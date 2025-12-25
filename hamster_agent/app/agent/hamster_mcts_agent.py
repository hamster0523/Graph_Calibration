import math
import random
from typing import List, Optional, Any, Dict, Tuple
from copy import deepcopy
import mmengine
import warnings
import json

warnings.simplefilter("ignore", DeprecationWarning)
from dataclasses import dataclass

from .toolcall import ToolCallAgent
from app.logger import logger
from app.schema import AgentState, Memory, Message, ToolChoice, ToolCall
from app.exceptions import TokenLimitExceeded
from app.agent.termination_policy import TerminationPolicy, TerminationOutcome, HotpotQATerminationPolicy

@dataclass
class MCTSConfig:
    max_depth: int = 5
    iterations: int = 5  # Total MCTS iterations
    n_generate_samples: int = 4  # Number of samples generated during expansion
    exploration_coef: float = 1.414  # UCB1 exploration coefficient (sqrt(2))
    negative_reward: float = -1.0  # Negative reward
    positive_reward: float = 1.0   # Positive reward
    simulation_depth: int = 5      # Maximum depth for simulation phase
    gamma: float = 0.95              # 折扣因子（用于simulation）
    rollout_coef: float = 1.0        # simulation回报的权重

class MCTSNode:
    def __init__(self, 
                 state: Optional[Memory] = None,
                 parent: Optional['MCTSNode'] = None,
                 action: Optional[str] = None,
                 depth: int = 0,
                 prior: float = 1.0,
                 visits: int = 0,
                 value: float = 0.0,
                 is_terminal: bool = False,
                 mcts_cfg: Optional[MCTSConfig] = None):
        """
        MCTS Node class
        
        Args:
            state: Current state (Memory object)
            parent: Parent node
            action: Action taken to reach this node
            depth: Node depth
            prior: Prior probability
            visits: Visit count
            value: Accumulated value
            is_terminal: Whether this is a terminal node
            mcts_cfg: MCTS configuration
        """
        self.state = state if state is not None else Memory()
        self.parent = parent
        self.action = action  # Action from parent to current node
        self.children: List['MCTSNode'] = []
        self.visits = visits
        self.value = value
        self.depth = depth
        self.is_terminal = is_terminal
        self.prior = prior
        self.mcts_cfg = mcts_cfg if mcts_cfg is not None else MCTSConfig()
        
        # Track possible actions
        self.untried_actions: List[str] = []
        self.expanded = False
        
        # New fields for reward separation and termination tracking
        self.edge_reward: float = 0.0  # Immediate step reward
        self.terminal_reward: float = 0.0  # Terminal reward when is_terminal=True
        self.termination_reason: str = ""  # Reason for termination
        self.success: Optional[bool] = None  # Whether the termination was successful
        
    def is_fully_expanded(self) -> bool:
        """Check if node is fully expanded"""
        return self.expanded and len(self.untried_actions) == 0
    
    def get_ucb_score(self) -> float:
        """Calculate UCB1 score for selection"""
        if self.visits == 0:
            return float('inf')  # Unvisited nodes have highest priority
        
        if self.parent is None:
            return self.value / self.visits
        
        exploitation = self.value / self.visits
        exploration = self.mcts_cfg.exploration_coef * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration
    
    def select_best_child(self) -> Optional['MCTSNode']:
        """Select best child node based on UCB1"""
        if not self.children:
            return None
        
        # Filter out terminal nodes
        valid_children = [child for child in self.children if not child.is_terminal]
        if not valid_children:
            self.is_terminal = True
            return None
        
        # Select child with highest UCB score
        best_child = max(valid_children, key=lambda child: child.get_ucb_score())
        return best_child
    
    def select_best_action_child(self) -> Optional['MCTSNode']:
        """Select child with most visits (final decision)"""
        if not self.children:
            return None
        
        valid_children = [child for child in self.children if not child.is_terminal]
        if not valid_children:
            return None
            
        return max(valid_children, key=lambda child: child.visits)
    
    def update(self, reward: float):
        """Update node statistics"""
        self.visits += 1
        self.value += reward
    
    def add_child(self, child_node: 'MCTSNode'):
        """Add child node"""
        child_node.parent = self
        self.children.append(child_node)
    
    def get_path_from_root(self) -> List['MCTSNode']:
        """Get path from root to current node"""
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))
    
    def get_action_sequence(self) -> List[str]:
        """Get action sequence from root to current node"""
        path = self.get_path_from_root()
        return [node.action for node in path[1:] if node.action is not None]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for saving, including full historical context"""
        # Convert Memory state to serializable format
        state_dict = None
        if self.state and hasattr(self.state, 'messages'):
            state_dict = {
                'messages': []
            }
            
            for msg in self.state.messages:
                msg_dict = {
                    'role': msg.role,
                    'content': msg.content,
                    'tool_call_id': getattr(msg, 'tool_call_id', None),
                    'name': getattr(msg, 'name', None)
                }
                
                # Handle tool_calls serialization
                tool_calls = getattr(msg, 'tool_calls', None)
                if tool_calls:
                    msg_dict['tool_calls'] = []
                    for tool_call in tool_calls:
                        if hasattr(tool_call, 'model_dump'):
                            # Pydantic model - use model_dump()
                            tool_call_dict = tool_call.model_dump()
                        elif hasattr(tool_call, 'dict'):
                            # Pydantic model - use dict()
                            tool_call_dict = tool_call.dict()
                        elif isinstance(tool_call, dict):
                            # Already a dictionary
                            tool_call_dict = tool_call
                        else:
                            # Fallback: convert to dict manually
                            tool_call_dict = {
                                'id': getattr(tool_call, 'id', ''),
                                'type': getattr(tool_call, 'type', 'function'),
                                'function': {
                                    'name': getattr(getattr(tool_call, 'function', None), 'name', ''),
                                    'arguments': getattr(getattr(tool_call, 'function', None), 'arguments', '')
                                }
                            }
                        msg_dict['tool_calls'].append(tool_call_dict)
                else:
                    msg_dict['tool_calls'] = None
                
                state_dict['messages'].append(msg_dict)
        
        return {
            'action': self.action,
            'visits': self.visits,
            'value': self.value,
            'depth': self.depth,
            'is_terminal': self.is_terminal,
            'prior': self.prior,
            'state': state_dict,  # Include full historical context
            'untried_actions': self.untried_actions,
            'expanded': self.expanded,
            'edge_reward': getattr(self, 'edge_reward', 0.0),
            'terminal_reward': getattr(self, 'terminal_reward', 0.0),
            'termination_reason': getattr(self, 'termination_reason', ''),
            'success': getattr(self, 'success', None),
            'children': [child.to_dict() for child in self.children],
            'ucb_score': self.get_ucb_score() if self.parent else 0,
            'action_sequence': self.get_action_sequence()
        }
    
class MCTSAgent(ToolCallAgent):
    """
    Standard MCTS agent implementing complete four-phase process:
    1. Selection: Select most promising leaf node
    2. Expansion: Expand new child nodes
    3. Simulation: Simulate to terminal state
    4. Backpropagation: Backpropagate reward values
    """
    
    def __init__(self, mcts_config: Optional[MCTSConfig] = None, termination_policy: Optional[TerminationPolicy] = None, **kwargs):
        super().__init__(**kwargs)
        self.mcts_config = mcts_config if mcts_config is not None else MCTSConfig()
        self.root: Optional[MCTSNode] = None
        self.all_paths: List[List[MCTSNode]] = []  # Save all explored paths
        self.now_think_content: str = ""
        
        # Set up termination policy - single decision point
        self.termination_policy = termination_policy if termination_policy is not None else TerminationPolicy()
        
        # MCTS special terminal tool names list, inheriting from parent class
        self.mcts_terminal_tools = self.special_tool_names.copy()
        logger.info(f"🔧 MCTS terminal tools list: {self.mcts_terminal_tools}")
        logger.info(f"🎯 Using termination policy: {type(self.termination_policy).__name__}")

    def update_termination_policy(self, new_termination_policy : TerminationPolicy):
        self.termination_policy = new_termination_policy
        #logger.info(f"Updated termination policy: {type(self.termination_policy).__name__}")

    def add_terminal_tool(self, tool_name: str):
        """
        Add custom terminal tool
        
        Args:
            tool_name: Tool name
        """
        if tool_name not in self.mcts_terminal_tools:
            self.mcts_terminal_tools.append(tool_name)
            logger.info(f"Added MCTS terminal tool: {tool_name}")
        else:
            logger.warning(f"Terminal tool already exists: {tool_name}")
    
    def remove_terminal_tool(self, tool_name: str):
        """
        Remove terminal tool
        
        Args:
            tool_name: Tool name
        """
        if tool_name in self.mcts_terminal_tools:
            self.mcts_terminal_tools.remove(tool_name)
            logger.info(f"Removed MCTS terminal tool: {tool_name}")
        else:
            logger.warning(f"Terminal tool does not exist: {tool_name}")
    
    def get_terminal_tools(self) -> List[str]:
        """Get current terminal tools list"""
        return self.mcts_terminal_tools.copy()
        
    async def search(self, query: str) -> Dict[str, Any]:
        """
        Execute MCTS search
        
        Args:
            query: Initial query
            
        Returns:
            Dictionary containing search results
        """
        # Initialize root node
        initial_memory = Memory()
        initial_memory.add_message(Message.user_message(query))
        self.root = MCTSNode(
            state=initial_memory,
            mcts_cfg=self.mcts_config
        )
        
        logger.info(f"🌳 Starting MCTS search, target iterations: {self.mcts_config.iterations}")
        
        # Execute MCTS iterations
        for iteration in range(self.mcts_config.iterations):
            logger.info(f"🔍 MCTS iteration {iteration + 1}/{self.mcts_config.iterations}")
            
            # MCTS four phases
            await self._mcts_iteration()
            
            # Check if solution found
            if self._has_solution():
                logger.info(f"✅ Solution found in iteration {iteration + 1}")
                break
        
        # Collect all paths
        self._collect_all_paths(self.root, [])
        
        # Return search results
        return {
            'best_path': self._get_best_path(),
            'all_paths': self.all_paths,
            'tree_stats': self._get_tree_statistics(),
            'root_node': self.root
        }
    
    async def _mcts_iteration(self):
        """Execute one complete MCTS iteration (four phases)"""
        
        if self.root is None:
            logger.error("❌ Root node is None, cannot execute MCTS iteration")
            return
        
        # 1️⃣ Selection phase
        selected_node = self._selection(self.root)
        if selected_node is None:
            logger.warning("⚠️ No valid node found in Selection phase")
            return
        
        # 2️⃣ Expansion phase
        expanded_node = await self._expansion(selected_node)
        if expanded_node is None:
            logger.warning("⚠️ Expansion phase failed")
            return
        
        # 3️⃣ Simulation phase
        reward = await self._simulation(expanded_node)
        
        # 4️⃣ Backpropagation phase
        self._backpropagation(expanded_node, reward)
    
    def _selection(self, node: MCTSNode) -> Optional[MCTSNode]:
        """
        Selection phase: From root node, use UCB1 to select to leaf node
        
        Args:
            node: Current node (usually root node)
            
        Returns:
            Selected leaf node
        """
        current = node
        path = [current]
        
        # Select downward along tree until finding unexpanded node
        while current.is_fully_expanded() and not current.is_terminal:
            current = current.select_best_child()
            if current is None:
                break
            path.append(current)
        
        logger.info(f"📍 Selection path length: {len(path)}, depth: {current.depth if current else 'None'}")
        return current
    
    async def _expansion(self, node: MCTSNode) -> Optional[MCTSNode]:
        """
        Expansion phase: Expand new child nodes on selected node
        
        Args:
            node: Node to expand
            
        Returns:
            Newly expanded child node
        """
        if node.is_terminal:
            return node
        
        # Check depth limit
        if node.depth >= self.mcts_config.max_depth:
            node.is_terminal = True
            return node
        
        try:
            # Generate possible actions
            possible_actions = await self._generate_actions(node)
            
            if not possible_actions:
                node.is_terminal = True
                return node
            
            # If there are untried actions, select one for expansion
            if not node.expanded:
                node.untried_actions = possible_actions
                node.expanded = True
            
            if node.untried_actions:
                # Select an action for expansion
                action = node.untried_actions.pop(0)
                child_node = await self._create_child_node(node, action)
                node.add_child(child_node)
                
                logger.info(f"🌱 Expanded new node: depth={child_node.depth}, action={action[:50]}...")
                return child_node
            
            # If all actions tried, select best child node
            return node.select_best_child()
            
        except Exception as e:
            logger.error(f"❌ Expansion phase error: {e}")
            node.is_terminal = True
            return node
        
    async def think(self) -> bool:
        """Process current state and decide next actions using tools"""
        if self.next_step_prompt:
            user_msg = Message.user_message(self.next_step_prompt)
            self.messages += [user_msg]

        try:
            # Get response with tool options
            response = await self.llm.ask_tool(
                messages=self.messages,
                system_msgs=(
                    [Message.system_message(self.system_prompt)]
                    if self.system_prompt
                    else None
                ),
                tools=self.available_tools.to_params(),
                tool_choice=self.tool_choices,
            )
        except ValueError:
            raise
        except Exception as e:
            # Check if this is a RetryError containing TokenLimitExceeded
            if hasattr(e, "__cause__") and isinstance(e.__cause__, TokenLimitExceeded):
                token_limit_error = e.__cause__
                logger.error(
                    f"🚨 Token limit error (from RetryError): {token_limit_error}"
                )
                self.memory.add_message(
                    Message.assistant_message(
                        f"Maximum token limit reached, cannot continue execution: {str(token_limit_error)}"
                    )
                )
                self.state = AgentState.FINISHED
                return False
            raise

        self.tool_calls = tool_calls = (
            response.tool_calls if response and response.tool_calls else []
        )
        content = response.content if response and response.content else ""

        # Log response info
        logger.info(f"✨ {self.name}'s thoughts: {content}")
        logger.info(
            f"🛠️ {self.name} selected {len(tool_calls) if tool_calls else 0} tools to use"
        )
        if tool_calls:
            logger.info(
                f"🧰 Tools being prepared: {[call.function.name for call in tool_calls]}"
            )
            logger.info(f"🔧 Tool arguments: {tool_calls[0].function.arguments}")

        try:
            if response is None:
                raise RuntimeError("No response received from the LLM")

            # Handle different tool_choices modes
            if self.tool_choices == ToolChoice.NONE:
                if tool_calls:
                    logger.warning(
                        f"🤔 Hmm, {self.name} tried to use tools when they weren't available!"
                    )
                if content:
                    self.now_think_content = content
                    return True
                return False

            # Create and add assistant message
            # assistant_msg = (
            #     Message.from_tool_calls(content=content, tool_calls=self.tool_calls)
            #     if self.tool_calls
            #     else Message.assistant_message(content)
            # )
            # self.memory.add_message(assistant_msg)
            self.now_think_content = content

            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                return True  # Will be handled in act()

            # For 'auto' mode, continue with content if no commands but content exists
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                return bool(content)

            return bool(self.tool_calls)
        except Exception as e:
            logger.error(f"🚨 Oops! The {self.name}'s thinking process hit a snag: {e}")
            # self.memory.add_message(
            #     Message.assistant_message(
            #         f"Error encountered while processing: {str(e)}"
            #     )
            # )
            self.now_think_content = f"Error encountered while processing: {str(e)}"
            return False
    
    async def _generate_actions(self, node: MCTSNode) -> List[str]:
        """
        Generate possible actions for given node
        
        Args:
            node: Current node
            
        Returns:
            List of possible actions
        """
        try:
            # Set current state
            self.memory = deepcopy(node.state)
            
            # Use parent class think method to generate actions
            should_act = await self.think()
            
            if not should_act or not self.tool_calls:
                return []
            
            # Extract actions from tool_calls
            actions = []
            for tool_call in self.tool_calls:
                action_str = f"{tool_call.function.name}({tool_call.function.arguments})"
                actions.append(action_str)
            
            # Limit number of generated actions
            return actions[:self.mcts_config.n_generate_samples]
            
        except Exception as e:
            logger.error(f"❌ Error generating actions: {e}")
            return []

    def _find_tool_call_by_name(self, name: str) -> Optional[ToolCall]:
        for tool_call in self.tool_calls:
            if tool_call.function.name == name:
                return tool_call

    async def _create_child_node(self, parent: MCTSNode, action: str) -> MCTSNode:
        """
        Create child node using unified termination strategy
        
        Args:
            parent: Parent node
            action: Action to execute
            
        Returns:
            New child node with proper reward tracking
        """
        try:
            # Copy parent node state
            new_state = deepcopy(parent.state)
            
            # Execute action and update state
            self.memory = new_state
            
            # Parse and execute action
            tool_name, args_str = self._parse_action(action)
            
            tool_call: Optional[ToolCall] = self._find_tool_call_by_name(tool_name)

            # Create and add assistant message
            assistant_msg = (
                Message.from_tool_calls(content=self.now_think_content, tool_calls=[tool_call])
                if tool_call
                else Message.assistant_message(self.now_think_content)
            )
            self.memory.add_message(assistant_msg)
            
            # Execute tool call
            result = ""
            if tool_call:
                result = await self.execute_tool(tool_call)
            
                tool_msg = Message.tool_message(
                    content=result,
                    tool_call_id=tool_call.id,
                    name=tool_name,
                    base64_image=getattr(self, '_current_base64_image', None),
                )
                self.memory.add_message(tool_msg)

            # Calculate immediate step reward (light shaping)
            edge_reward = self._evaluate_step_result(result, tool_name)
            
            # Use unified termination strategy
            outcome = self._judge_termination(tool_name, result, parent.depth + 1, self.memory)
            
            # Create child node
            child_node = MCTSNode(
                state=deepcopy(self.memory),
                parent=parent,
                action=action,
                depth=parent.depth + 1,
                is_terminal=outcome.terminated,
                mcts_cfg=self.mcts_config
            )
            
            # Store rewards on node (don't modify visits/value here)
            child_node.edge_reward = edge_reward
            child_node.terminal_reward = outcome.reward if outcome.terminated else 0.0
            child_node.termination_reason = outcome.reason
            child_node.success = outcome.success
            
            logger.info(f"Child node created at depth {child_node.depth}: "
                       f"edge_reward={edge_reward:.3f}, "
                       f"terminal_reward={child_node.terminal_reward:.3f}, "
                       f"terminated={outcome.terminated}, "
                       f"reason={outcome.reason}")
            
            return child_node
            
        except Exception as e:
            logger.error(f"❌ Error creating child node: {e}")
            # Return terminal node with error
            error_node = MCTSNode(
                state=deepcopy(parent.state),
                parent=parent,
                action=action,
                depth=parent.depth + 1,
                is_terminal=True,
                mcts_cfg=self.mcts_config
            )
            error_node.edge_reward = 0.0
            error_node.terminal_reward = 0.0
            error_node.termination_reason = "error"
            error_node.success = False
            return error_node
    
    def _parse_action(self, action: str) -> tuple[str, str]:
        """Parse action string"""
        try:
            if '(' in action and action.endswith(')'):
                tool_name = action[:action.index('(')]
                args_str = action[action.index('(') + 1:-1]
                return tool_name, args_str
            else:
                return action, "{}"
        except:
            return "unknown_action", "{}"
    
    def _is_mcts_terminal_tool(self, tool_name: str) -> bool:
        """Check if it's an MCTS terminal tool"""
        return tool_name.lower() in [name.lower() for name in self.mcts_terminal_tools]
    
    def _judge_termination(self, tool_name: Optional[str], result: str, depth: int, state: Memory) -> TerminationOutcome:
        """
        Unified termination judgment using strategy pattern
        
        Args:
            tool_name: Name of the tool used (if any)
            result: Tool execution result  
            depth: Current node depth
            state: Current memory state
            
        Returns:
            TerminationOutcome with termination decision and reward
        """
        return self.termination_policy.decide(
            tool_name=tool_name,
            result=result,
            depth=depth,
            max_depth=self.mcts_config.max_depth,
            state=state
        )
    
    def _evaluate_step_result(self, result: str, tool_name: Optional[str] = None) -> float:
        """
        Evaluate immediate step reward (edge reward)
        
        Args:
            result: Tool execution result
            tool_name: Tool name (optional)
            
        Returns:
            Edge reward in range [0, 0.2] for reward shaping
        """
        if not result:
            return 0.0
        
        reward = 0.0
        result_lower = result.lower()
        
        # Positive indicators (small rewards for shaping)
        if any(keyword in result_lower for keyword in ["found", "discovered", "evidence", "information"]):
            reward += 0.05
        if any(keyword in result_lower for keyword in ["relevant", "helpful", "useful"]):
            reward += 0.03
        if len(result) > 100:  # Substantial response
            reward += 0.02
            
        # Negative indicators
        if any(keyword in result_lower for keyword in ["not found", "no results", "unavailable"]):
            reward -= 0.03
        if any(keyword in result_lower for keyword in ["error", "failed", "invalid"]):
            reward -= 0.05
            
        # Clamp to range [0, 0.2] for controlled shaping
        return max(0.0, min(0.2, reward))
    
    async def _simulation(self, node: MCTSNode) -> float:
        """
        Simulation phase: 从扩展的子节点开始进行rollout，真正前进状态直到终止条件
        
        Args:
            node: Current node (刚扩展出的子节点)
            
        Returns:
            Simulated reward value (normalized to [0,1])
        """
        # If terminal node, return stored rewards
        if node.is_terminal:
            terminal_reward = getattr(node, "terminal_reward", 0.0)
            edge_reward = getattr(node, "edge_reward", 0.0)
            return terminal_reward + edge_reward
        
        current_depth = node.depth
        sim_memory = deepcopy(node.state)
        self.memory = sim_memory
        
        R = 0.0
        gamma = self.mcts_config.gamma
        
        try:
            for t in range(self.mcts_config.simulation_depth):
                if current_depth >= self.mcts_config.max_depth:
                    break
                
                should_act = await self.think()
                if not should_act or not self.tool_calls:
                    break
                
                # 随机选择一个 tool_call
                tool_call = random.choice(self.tool_calls)
                
                # 1) 记录 assistant 的"思考/调用"
                assistant_msg = Message.from_tool_calls(
                    content=self.now_think_content, tool_calls=[tool_call]
                )
                self.memory.add_message(assistant_msg)
                
                # 2) 执行工具并记录 tool 消息
                result = await self.execute_tool(tool_call)
                tool_msg = Message.tool_message(
                    content=result,
                    tool_call_id=tool_call.id,
                    name=tool_call.function.name,
                    base64_image=getattr(self, '_current_base64_image', None),
                )
                self.memory.add_message(tool_msg)
                
                tool_name = tool_call.function.name
                step_edge = self._evaluate_step_result(result, tool_name)
                outcome = self._judge_termination(tool_name, result, current_depth + 1, self.memory)
                
                # 3) 折扣累计
                step_total = step_edge + (outcome.reward if outcome.terminated else 0.0)
                R += (gamma ** t) * step_total
                
                if outcome.terminated:
                    logger.debug(f"🎯 Simulation terminated at step {t+1}, depth {current_depth + 1}: "
                               f"reason={outcome.reason}, step_reward={step_total:.3f}")
                    break
                
                current_depth += 1
            
            # Normalize final reward
            R = max(0.0, min(1.0, R))
            logger.debug(f"🎲 Simulation completed: {t + 1} steps, total reward: {R:.3f}")
            return R
            
        except Exception as e:
            logger.error(f"❌ Simulation phase error: {e}")
            return 0.0
    
    # def _evaluate_terminal_node(self, node: MCTSNode) -> float:
    #     """Evaluate terminal node value"""
    #     # If node already has preset value (like special terminal tool), return directly
    #     if node.visits > 0 and node.value != 0:
    #         return node.value / node.visits
        
    #     # Depth-based reward
    #     if node.depth >= self.mcts_config.max_depth:
    #         return self.mcts_config.negative_reward
        
    #     # Check if last action is special terminal tool
    #     if node.action:
    #         tool_name, _ = self._parse_action(node.action)
    #         if self._is_mcts_terminal_tool(tool_name):
    #             # Special terminal tool reward already set during creation, give base reward here
    #             if tool_name.lower() == "terminate":
    #                 return self.mcts_config.positive_reward
    #             elif tool_name.lower() == "reflection_terminate":
    #                 return self.mcts_config.positive_reward * 1.2
    #             else:
    #                 return self.mcts_config.positive_reward * 0.8
        
    #     # Check last message content
    #     if node.state.messages:
    #         last_message = node.state.messages[-1]
    #         if hasattr(last_message, 'content') and last_message.content:
    #             content_lower = last_message.content.lower()
    #             if any(success_word in content_lower for success_word in ['completed', 'success', 'finished']):
    #                 return self.mcts_config.positive_reward * 0.5
    #             elif any(fail_word in content_lower for fail_word in ['failed', 'error', 'cannot']):
    #                 return self.mcts_config.negative_reward * 0.5
        
    #     return 0.0
    
    def _backpropagation(self, node: MCTSNode, rollout_reward: float):
        """
        Backpropagation phase: Backpropagate reward values along path
        Now properly incorporates simulation/rollout rewards
        
        Args:
            node: Starting node for backpropagation
            rollout_reward: Reward from simulation/rollout phase
        """
        # Calculate total reward for this step
        step_reward = getattr(node, "edge_reward", 0.0)
        if node.is_terminal:
            step_reward += getattr(node, "terminal_reward", 0.0)
        
        # 加入 rollout 回报（使用配置的权重）
        total_reward = step_reward + self.mcts_config.rollout_coef * rollout_reward
        
        # Ensure reward is normalized
        total_reward = max(0.0, min(1.0, total_reward))
        
        current = node
        step = 0
        
        while current is not None:
            current.update(total_reward)
            
            # Debug logging for starting node only
            if step == 0:
                logger.debug(f"📊 Backprop at depth {current.depth}: "
                           f"edge_reward={getattr(current, 'edge_reward', 0.0):.3f}, "
                           f"terminal_reward={getattr(current, 'terminal_reward', 0.0):.3f}, "
                           f"rollout_reward={rollout_reward:.3f}, "
                           f"total_reward={total_reward:.3f}, "
                           f"visits={current.visits}, "
                           f"avg_value={current.value/current.visits if current.visits > 0 else 0:.3f}")
            
            current = current.parent
            step += 1
        
        logger.debug(f"📈 Backpropagation: Updated {step} nodes, total reward: {total_reward:.3f}")
    
    def _has_solution(self) -> bool:
        """Check if solution found"""
        if not self.root or not self.root.children:
            return False
        
        # Check if there are high-value child nodes
        for child in self.root.children:
            if child.visits > 0 and (child.value / child.visits) > 0.5:
                return True
        
        return False
    
    def _collect_all_paths(self, node: MCTSNode, current_path: List[MCTSNode]):
        """Collect all explored paths"""
        current_path.append(node)
        
        if node.is_terminal or not node.children:
            # Save complete path
            self.all_paths.append(current_path.copy())
        else:
            # Recursively collect sub-paths
            for child in node.children:
                if child.visits > 0:  # Only save visited paths
                    self._collect_all_paths(child, current_path.copy())
        
        current_path.pop()
    
    def _get_best_path(self) -> Optional[List[MCTSNode]]:
        """Get best path"""
        if not self.root:
            return None
        
        path = [self.root]
        current = self.root
        
        while current.children:
            best_child = current.select_best_action_child()
            if best_child is None:
                break
            path.append(best_child)
            current = best_child
        
        return path
    
    def _get_tree_statistics(self) -> Dict[str, Any]:
        """Get search tree statistics"""
        if not self.root:
            return {}
        
        def collect_stats(node: MCTSNode, stats: Dict[str, Any]):
            stats['total_nodes'] += 1
            stats['total_visits'] += node.visits
            
            if node.is_terminal:
                stats['terminal_nodes'] += 1
            
            if node.visits > 0:
                stats['visited_nodes'] += 1
            
            stats['max_depth'] = max(stats['max_depth'], node.depth)
            
            for child in node.children:
                collect_stats(child, stats)
        
        stats = {
            'total_nodes': 0,
            'visited_nodes': 0,
            'terminal_nodes': 0,
            'total_visits': 0,
            'max_depth': 0,
            'total_paths': len(self.all_paths)
        }
        
        collect_stats(self.root, stats)
        return stats
    
    def save(self, output_path: str):
        """Save MCTS search results"""
        if not self.root:
            logger.warning("⚠️ No search results to save")
            return
        
        result = {
            'mcts_config': {
                'max_depth': self.mcts_config.max_depth,
                'iterations': self.mcts_config.iterations,
                'n_generate_samples': self.mcts_config.n_generate_samples,
                'exploration_coef': self.mcts_config.exploration_coef
            },
            'tree': self.root.to_dict(),
            'statistics': self._get_tree_statistics(),
            'all_paths_count': len(self.all_paths),
            'best_path_actions': [node.action for node in (self._get_best_path() or [])[1:] if node.action]
        }
        
        mmengine.dump(result, output_path, ensure_ascii=False, indent=2)
        logger.info(f"💾 MCTS search results saved to: {output_path}")

    def save_jsonl(self, output_path: str):
        """Append MCTS search result as one JSON object per line"""
        if not self.root:
            logger.warning("⚠️ No search results to save")
            return
        
        result = {
            'mcts_config': {
                'max_depth': self.mcts_config.max_depth,
                'iterations': self.mcts_config.iterations,
                'n_generate_samples': self.mcts_config.n_generate_samples,
                'exploration_coef': self.mcts_config.exploration_coef
            },
            'tree': self.root.to_dict(),
            'statistics': self._get_tree_statistics(),
            'all_paths_count': len(self.all_paths),
            'best_path_actions': [node.action for node in (self._get_best_path() or [])[1:] if node.action]
        }

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n") 
    
    @staticmethod
    def load(data_path: str) -> Optional[MCTSNode]:
        """Load MCTS search results from file"""
        try:
            data = mmengine.load(data_path)
            
            def dict_to_node(node_data: Dict[str, Any], parent: Optional[MCTSNode] = None) -> MCTSNode:
                """Recursively build nodes from dictionary"""
                children_data = node_data.pop('children', [])
                
                # Restore Memory state from saved data
                state = Memory()
                if 'state' in node_data and node_data['state']:
                    state_data = node_data['state']
                    if 'messages' in state_data:
                        for msg_data in state_data['messages']:
                            message = Message(
                                role=msg_data['role'],
                                content=msg_data['content']
                            )
                            
                            # Restore tool_calls if they exist
                            if msg_data.get('tool_calls'):
                                from app.schema import ToolCall, Function
                                tool_calls = []
                                for tc_data in msg_data['tool_calls']:
                                    # Reconstruct ToolCall object from dictionary
                                    if isinstance(tc_data, dict):
                                        function_data = tc_data.get('function', {})
                                        function = Function(
                                            name=function_data.get('name', ''),
                                            arguments=function_data.get('arguments', '')
                                        )
                                        tool_call = ToolCall(
                                            id=tc_data.get('id', ''),
                                            type=tc_data.get('type', 'function'),
                                            function=function
                                        )
                                        tool_calls.append(tool_call)
                                    else:
                                        # Fallback: assume it's already a ToolCall object
                                        tool_calls.append(tc_data)
                                message.tool_calls = tool_calls
                            
                            # Restore other message attributes
                            if msg_data.get('tool_call_id'):
                                message.tool_call_id = msg_data['tool_call_id']
                            if msg_data.get('name'):
                                message.name = msg_data['name']
                                
                            state.add_message(message)
                
                # Create node with restored state and additional properties
                node_args = {
                    'state': state,
                    'depth': node_data.get('depth', 0),
                    'visits': node_data.get('visits', 0),
                    'value': node_data.get('value', 0.0),
                    'is_terminal': node_data.get('is_terminal', False),
                    'prior': node_data.get('prior', 1.0),
                    'parent': parent
                }
                
                if 'action' in node_data:
                    node_args['action'] = node_data['action']
                
                node = MCTSNode(**node_args)
                
                # Restore additional node properties
                node.untried_actions = node_data.get('untried_actions', [])
                node.expanded = node_data.get('expanded', False)
                
                # Restore reward tracking fields
                node.edge_reward = node_data.get('edge_reward', 0.0)
                node.terminal_reward = node_data.get('terminal_reward', 0.0)
                node.termination_reason = node_data.get('termination_reason', '')
                node.success = node_data.get('success', None)
                
                # Recursively build child nodes
                for child_data in children_data:
                    child = dict_to_node(child_data, node)
                    node.children.append(child)
                
                return node
            
            if 'tree' in data:
                return dict_to_node(data['tree'])
            else:
                # Compatible with old format
                return dict_to_node(data)
                
        except Exception as e:
            logger.error(f"❌ Failed to load MCTS data: {e}")
            return None
    
    def load_from_file(self, data_path: str) -> bool:
        """Load MCTS tree from file"""
        loaded_root = self.load(data_path)
        if loaded_root:
            self.root = loaded_root
            logger.info(f"✅ Successfully loaded MCTS tree from: {data_path}")
            return True
        return False
    
    async def run_mcts_search(self, query: str, save_path: Optional[str] = None) -> str:
        """
        Run complete MCTS search and return results
        
        Args:
            query: Search query
            save_path: Optional save path
            
        Returns:
            Formatted search result string
        """
        # logger.info(f"🚀 Starting MCTS search: {query}")
        
        # Execute search
        search_result = await self.search(query)
        
        # Save results
        if save_path:
            self.save(save_path)
        
        # Format output
        best_path = search_result['best_path']
        stats = search_result['tree_stats']
        
        result_summary = f"""
        🌳 MCTS search completed!

        📊 Search statistics:
        - Total nodes: {stats.get('total_nodes', 0)}
        - Visited nodes: {stats.get('visited_nodes', 0)}
        - Terminal nodes: {stats.get('terminal_nodes', 0)}
        - Total visits: {stats.get('total_visits', 0)}
        - Max depth: {stats.get('max_depth', 0)}
        - Total paths: {stats.get('total_paths', 0)}

        🎯 Best path ({len(best_path) if best_path else 0} steps):
        """
        
        if best_path:
            for i, node in enumerate(best_path):
                if node.action:
                    result_summary += f"  {i}. {node.action}\n"
                result_summary += f"     Visits: {node.visits}, Avg value: {node.value/node.visits if node.visits > 0 else 0:.3f}\n"
        
        result_summary += f"\n🗂️ Total saved paths: {len(self.all_paths)} (including all explored suboptimal paths)"
        
        return result_summary

from app.prompt.hotpotqa_agent import HOT_SYSTEM_PROMPT, HOT_NEXT_STEP_PROMPT

class MCTSAgent_HotpotQA(MCTSAgent):
    
    system_prompt: str = HOT_SYSTEM_PROMPT
    next_step_prompt: str = HOT_NEXT_STEP_PROMPT

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def run_mcts_search(self, query: str, save_path: Optional[str] = None):
        
        search_result = await self.search(query)
        
        # Save results override for save to jsonl
        if save_path:
            self.save_jsonl(save_path)
        
        return
    
    def _has_solution(self) -> bool:
        def dfs(node: MCTSNode) -> bool:
            if node.is_terminal:
                return True
            return any(dfs(ch) for ch in node.children)
        return dfs(self.root) if self.root else False

    
    # def _has_solution(self) -> bool:
    #     def dfs(node : MCTSNode):
    #         if node.is_terminal and (node.terminal_reward > 0 or node.success):
    #             return True
    #         for child in node.children:
    #             if dfs(child):
    #                 return True
    #         return False
        
    #     return dfs(self.root) if self.root else False


        
    