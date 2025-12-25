import os
import json
import pandas as pd
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import asyncio
from datetime import datetime
from tqdm.asyncio import tqdm

from app.exceptions import ToolError
from app.tool.base import BaseTool, ToolResult
from app.llm import LLM
from app.logger import logger

# Gaming performance metric categories
COLUMN_CATEGORIES = {
        "fps": ["fps", "frame", "framerate", "frame_rate", "framecount", "frame_count", "frametime", "frame_time"],
        "memory": ["memory", "mem", "ram", "heap", "allocation", "gc", "garbage", "alloc"],
        "cpu": ["cpu", "processor", "core", "thread", "utilization", "usage"],
        "gpu": ["gpu", "graphic", "render", "rendering", "shader", "draw", "pipeline", "vertex", "texture"],
        "network": ["network", "net", "ping", "latency", "packet", "bandwidth", "connection", "http", "socket"],
        "storage": ["storage", "disk", "io", "read", "write", "load", "save", "file"],
        "battery": ["battery", "power", "energy", "consumption", "drain"],
        "thermal": ["thermal", "temperature", "temp", "heat", "cooling", "throttle"],
        "audio": ["audio", "sound", "voice", "music", "mix", "volume"],
        "input": ["input", "touch", "controller", "keyboard", "mouse", "button", "click", "press"],
        "asset": ["asset", "resource", "bundle", "model", "mesh", "animation"],
        "physics": ["physics", "collision", "rigid", "body", "raycast", "simulation"],
        "ai": ["ai", "pathfinding", "navigation", "behavior", "decision"],
        "ui": ["ui", "interface", "button", "panel", "menu", "hud", "widget"],
        "scene": ["scene", "level", "world", "environment", "lighting", "camera"],
        "system": ["system", "os", "device", "hardware", "software", "version", "build", "platform"],
        "time": ["time", "timestamp", "date", "duration", "elapsed", "timer"],
        "event": ["event", "callback", "trigger", "notification", "signal"],
        "error": ["error", "exception", "crash", "bug", "fault", "failure", "warning"],
        "other": [],  # Catch-all category
    }


class ExcelColumnAnalysisTool(BaseTool):
    """
    A tool for analyzing column names in game performance CSV files.
    
    This tool analyzes CSV files containing game engine performance data across different
    versions and devices. It categorizes column names, provides detailed analysis of each column's
    potential meaning, and generates a JSON metadata file for further analysis.
    """

    name: str = "excel_column_analysis_tool"
    description: str = "Analyze and categorize columns in game performance CSV files, generating metadata for further analysis."
    parameters: dict = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the CSV file containing game performance data",
            },
            "output_path": {
                "type": "string",
                "description": "Optional path for the output JSON file. If not provided, will be saved alongside the CSV file.",
            },
            "encoding": {
                "type": "string",
                "description": "File encoding (default: auto-detect)",
            },
            "sample_rows": {
                "type": "integer",
                "description": "Number of rows to sample for analysis (default: 100)",
                "default": 100,
            },
            "max_concurrency": {
                "type": "integer",
                "description": "Maximum number of concurrent LLM calls (default: 5)",
                "default": 5,
            },
        },
        "required": ["file_path"],
    }

    async def execute(
        self,
        file_path: str,
        output_path: Optional[str] = None,
        encoding: Optional[str] = None,
        sample_rows: int = 100,
        max_concurrency: int = 5,
    ) -> str:
        """
        Analyze columns in a game performance CSV file and generate a JSON metadata file.
        
        Args:
            file_path: Path to the CSV file to analyze
            output_path: Optional path for the output JSON file
            encoding: File encoding (auto-detect if not specified)
            sample_rows: Number of rows to sample for analysis
            max_concurrency: Maximum number of concurrent LLM calls
            
        Returns:
            str: A summary of the analysis results
        """
        try:
            start_time = datetime.now()
            
            # Validate file path
            if not os.path.exists(file_path):
                return ToolError(f"File not found: {file_path}")
                
            # Generate output path if not provided
            if not output_path:
                file_base, _ = os.path.splitext(file_path)
                output_path = f"{file_base}_column_analysis.json"
                
            # Read the CSV file
            df, used_encoding = self._read_csv_with_encoding(file_path, encoding)
            
            # Basic file stats
            total_rows = len(df)
            total_columns = len(df.columns)
            
            logger.info(f"Analyzing {total_columns} columns from {file_path}")
            
            # Analyze column names and categorize them
            analysis_result = await self._analyze_columns(df, sample_rows, max_concurrency)
            
            # Add metadata
            metadata = {
                "source_file": file_path,
                "analysis_timestamp": datetime.now().isoformat(),
                "encoding_used": used_encoding,
                "total_rows": total_rows,
                "total_columns": total_columns,
                "sample_rows_analyzed": min(sample_rows, total_rows)
            }
            
            # Combine metadata and analysis results
            full_result = {
                "analysis_metadata": metadata,
                "columns_analysis": analysis_result
            }
            
            # Ensure the output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Created directory: {output_dir}")
            
            # Save to JSON file
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(full_result, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved analysis results to: {output_path}")
            except Exception as write_error:
                logger.error(f"Error writing to file {output_path}: {str(write_error)}")
                # Try to save to a fallback location if primary save fails
                fallback_path = os.path.join(os.getcwd(), "column_analysis_result.json")
                logger.info(f"Attempting to save to fallback location: {fallback_path}")
                with open(fallback_path, 'w', encoding='utf-8') as f:
                    json.dump(full_result, f, indent=2, ensure_ascii=False)
                output_path = fallback_path
                
            # Generate category statistics
            category_counts = Counter([col_data["refined_category"] for col_data in analysis_result.values()])
            
            # Calculate elapsed time
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            # Generate result message
            result_message = self._generate_result_message(
                file_path, 
                output_path,
                total_columns,
                total_rows,
                category_counts,
                elapsed_time
            )
            
            return ToolResult(output=result_message)
            
        except ToolError as e:
            logger.error(f"Error in ExcelColumnAnalysisTool: {str(e)}", exc_info=True)
            return  ToolError(f"Failed to analyze columns: {str(e)}")
        except Exception as e:
            logger.error(f"Error in ExcelColumnAnalysisTool: {str(e)}", exc_info=True)
            return  ToolError(f"Failed to analyze columns: {str(e)}")
            
    def _read_csv_with_encoding(self, file_path: str, encoding: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
        """Read CSV file with the specified or auto-detected encoding"""
        # Try specified encoding first
        if encoding:
            try:
                return pd.read_csv(file_path, encoding=encoding), encoding
            except Exception as e:
                logger.warning(f"Failed to read with specified encoding {encoding}: {str(e)}")
                
        # Auto-detect encoding
        encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        
        for enc in encodings_to_try:
            try:
                return pd.read_csv(file_path, encoding=enc), enc
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Error reading CSV with {enc} encoding: {str(e)}")
                
        # Last resort: try with the default encoding and errors='replace'
        try:
            return pd.read_csv(file_path, encoding='utf-8', errors='replace'), 'utf-8 (with replacements)'
        except Exception as e:
            raise ToolError(f"Failed to read CSV file with any encoding: {str(e)}")
            
    async def _analyze_columns(self, df: pd.DataFrame, sample_rows: int, max_concurrency: int) -> Dict[str, Any]:
        """Analyze all columns in the dataframe with progress bar"""
        # Create a semaphore to limit concurrent LLM calls
        semaphore = asyncio.Semaphore(max_concurrency)
        
        # First, get a refined categorization system using LLM
        logger.info("Generating refined category system from column names...")
        refined_categories = await self._generate_refined_categories(df.columns)
        
        # Create tasks for analyzing each column
        tasks = []
        for column_name in df.columns:
            task = self._analyze_column_with_semaphore(
                semaphore, 
                column_name, 
                df[column_name], 
                df[column_name].head(sample_rows),
                refined_categories
            )
            tasks.append(task)
            
        # Execute all tasks concurrently with progress bar and collect results
        logger.info(f"Analyzing {len(df.columns)} columns...")
        column_analyses = {}
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Analyzing columns"):
            column_name, analysis = await future
            column_analyses[column_name] = analysis
        
        return column_analyses
        
    async def _analyze_column_with_semaphore(
        self, 
        semaphore: asyncio.Semaphore, 
        column_name: str, 
        column_data: pd.Series, 
        sample_data: pd.Series,
        refined_categories: Dict[str, Dict[str, List[str]]]
    ) -> Tuple[str, Dict[str, Any]]:
        """Analyze a single column with semaphore to limit concurrency"""
        async with semaphore:
            return column_name, await self._analyze_column(column_name, column_data, sample_data, refined_categories)
            
    async def _analyze_column(
        self, 
        column_name: str, 
        column_data: pd.Series, 
        sample_data: pd.Series,
        refined_categories: Dict[str, Dict[str, List[str]]]
    ) -> Dict[str, Any]:
        """Analyze a single column and return its properties"""
        # Basic column statistics
        total_count = len(column_data)
        non_null_count = column_data.count()
        null_count = total_count - non_null_count
        
        try:
            unique_count = column_data.nunique()
            duplicate_count = total_count - unique_count if unique_count > 0 else total_count
        except:
            # For columns that can't be compared (e.g., lists, dicts)
            unique_count = "N/A"
            duplicate_count = "N/A"
            
        # Get data type
        data_type = str(column_data.dtype)
        
        # Sample values (non-null)
        sample_values = sample_data.dropna().tolist()[:5]  # Up to 5 non-null samples
        
        # Most common values (if applicable)
        most_common_values = {}
        if hasattr(column_data, 'value_counts') and callable(column_data.value_counts):
            try:
                value_counts = column_data.value_counts().head(5).to_dict()
                most_common_values = {str(k): int(v) for k, v in value_counts.items()}
            except:
                pass
                
        # First use basic categorization method
        basic_category = self._infer_column_category(column_name)
        
        # Then use the refined categorization system
        refined_category, refined_subcategory = self._match_to_refined_categories(
            column_name, basic_category, refined_categories
        )
        
        # Generate reasoning about the column's purpose using LLM
        category_reasoning = await self._generate_category_reasoning(
            column_name, refined_category, refined_subcategory, sample_values
        )
        
        # Compile the analysis
        analysis = {
            "column_name": column_name,
            "data_type": data_type,
            "total_count": total_count,
            "non_null_count": str(non_null_count),  # Convert to string to ensure JSON serialization
            "null_count": str(null_count),
            "unique_count": unique_count,
            "duplicate_count": duplicate_count,
            "sample_values": sample_values,
            "most_common_values": most_common_values,
            "basic_category": basic_category,
            "refined_category": refined_category,
            "refined_subcategory": refined_subcategory,
            "category_reasoning": category_reasoning
        }
        
        return analysis
        
    def _infer_column_category(self, column_name: str) -> str:
        """Infer the category of a column based on its name"""
        column_lower = column_name.lower()
        
        # Check each category
        for category, keywords in COLUMN_CATEGORIES.items():
            for keyword in keywords:
                if keyword in column_lower:
                    return category
                    
        # Check for common patterns
        if re.search(r'(fps|frame|framerate|frame_rate)', column_lower):
            return "fps"
        elif re.search(r'(mem|memory|heap|alloc|gc)', column_lower):
            return "memory"
        elif re.search(r'(cpu|thread|core|processor)', column_lower):
            return "cpu"
        elif re.search(r'(gpu|render|shader|draw)', column_lower):
            return "gpu"
        elif re.search(r'(net|ping|latency|packet)', column_lower):
            return "network"
        elif re.search(r'(load|time|duration)', column_lower):
            return "time"
        elif re.search(r'(temp|thermal|heat)', column_lower):
            return "thermal"
        elif re.search(r'(device|hardware|model|platform)', column_lower):
            return "system"
        elif re.search(r'(error|crash|exception)', column_lower):
            return "error"
            
        # Default to "other" if no match
        return "other"
        
    async def _generate_category_reasoning(
        self, 
        column_name: str, 
        main_category: str,
        subcategory: str, 
        sample_values: List = None
    ) -> str:
        """Generate reasoning about why a column belongs to its category using LLM"""
        try:
            # Prepare sample data
            sample_data_str = ""
            if sample_values and len(sample_values) > 0:
                sample_data_str = f"Sample values: {', '.join(str(v) for v in sample_values[:5])}"
                
            # Construct prompt for LLM
            prompt = f"""You are an expert in analyzing game performance data. Please analyze this CSV column and provide reasoning for its categorization:

Column Name: "{column_name}"
Main Category: "{main_category}"
Subcategory: "{subcategory}"
{sample_data_str}

In 50-100 words, explain why this column likely belongs to the assigned category and subcategory, focusing on:
1. The relevance of the column name to the category and subcategory
2. What the data pattern suggests about its purpose in game performance analysis
3. Any key insights from the sample values that are useful for game performance optimization

Please keep the explanation concise, professional, and specific to the domain of game performance data analysis.
"""

            # Call LLM
            llm = LLM(config_name="default")
            messages = [{"role": "user", "content": prompt}]
            
            reasoning = await llm.ask(
                messages=messages,
                temperature=0.7,
                stream=False
            )
            
            return reasoning.strip()
            
        except Exception as e:
            logger.warning(f"Failed to generate reasoning for column {column_name}: {str(e)}")
            return f"This column appears to be related to {main_category} ({subcategory}) based on its name."
            
    def _generate_result_message(
        self, 
        file_path: str, 
        output_path: str,
        total_columns: int,
        total_rows: int,
        category_counts: Counter,
        elapsed_time: float
    ) -> str:
        """Generate a summary message of the analysis results"""
        
        # Format category counts for display
        categories_summary = "\n".join([
            f"• {category}: {count} columns ({count/total_columns*100:.1f}%)"
            for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        ])
        
        message = [
            "✅ Column Analysis Complete!",
            "",
            "📊 Analysis Summary:",
            f"• Source File: {file_path}",
            f"• Total Rows: {total_rows}",
            f"• Total Columns: {total_columns}",
            f"• Analysis Time: {elapsed_time:.2f} seconds",
            "",
            "🔍 Category Distribution:",
            categories_summary,
            "",
            "📝 Results saved to:",
            f"{output_path}",
            "",
            "The JSON file contains detailed analysis of each column, including:",
            "• Data type and statistics (null count, unique values, etc.)",
            "• Sample values and most common values",
            "• Basic and refined categorization (main categories and subcategories)",
            "• AI-generated reasoning explaining the categorization",
            "",
            "This metadata can now be used by other agents for detailed performance analysis."
        ]
        
        return "\n".join(message)
    
    async def _generate_refined_categories(self, column_names: List[str]) -> Dict[str, Dict[str, List[str]]]:
        """
        Generate a refined categorization system for column names using LLM
        
        This method asks the LLM to create a hierarchical categorization system
        specifically tailored to the column names in the dataset.
        
        Args:
            column_names: List of column names to categorize
            
        Returns:
            A dictionary of refined categories with subcategories and keywords
        """
        try:
            # Create a sample of column names (up to 100) to send to LLM
            # Using a sample because there might be hundreds or thousands of columns
            sample_size = min(100, len(column_names))
            column_sample = list(column_names)[:sample_size]
            
            # Create prompt for LLM
            column_names_str = "\n".join([f"- {col}" for col in column_sample])
            
            prompt = f"""You are an expert in game performance data analysis. I need your help to create a detailed, hierarchical categorization system for the columns in a game performance dataset.

Here is a sample of the column names from the dataset:

{column_names_str}

Based on these column names, please create a hierarchical categorization system with the following structure:
1. Main categories (e.g., "fps", "memory", "network", etc.)
2. Subcategories under each main category
3. For each subcategory, a list of keywords that can be used to identify column names that belong to it

Please follow these guidelines:
- Create 15-25 main categories covering all aspects of game performance
- Each main category should have 1-2 subcategories
- Each subcategory should have 3-10 keywords that can be used to match column names
- Ensure the categorization is as comprehensive as possible
- Be specific to game performance analysis terminology

FORMAT YOUR RESPONSE AS A JSON OBJECT with the following structure:
```json
{{
  "main_category_1": {{
    "subcategory_1": ["keyword1", "keyword2", "keyword3"],
    "subcategory_2": ["keyword4", "keyword5", "keyword6"]
  }},
  "main_category_2": {{
    "subcategory_3": ["keyword7", "keyword8", "keyword9"],
    "subcategory_4": ["keyword10", "keyword11", "keyword12"]
  }}
}}
```

For example, a partial response might look like:
```json
{{
  "fps": {{
    "frame_rate": ["fps", "framerate", "frame_rate", "frames_per_second"],
    "frame_time": ["frametime", "frame_time", "frame_latency", "rendertime"],
    "frame_pacing": ["framepacing", "frame_pacing", "jank", "stutter", "stability"]
  }},
  "memory": {{
    "usage": ["memory", "mem_usage", "ram", "memory_usage"],
    "allocation": ["alloc", "allocation", "memory_alloc", "heap"],
    "garbage_collection": ["gc", "garbage", "gctime", "collection"]
  }}
}}
```

ONLY RETURN THE JSON OBJECT, with no other text before or after.
"""

            # Call LLM
            llm = LLM(config_name="default")
            messages = [{"role": "user", "content": prompt}]
            
            response = await llm.ask(
                messages=messages,
                temperature=0.7
            )
            
            # Parse the JSON response
            # Find the JSON content between triple backticks if present
            json_str = response
            if "```json" in response and "```" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
                
            # Strip any leading/trailing whitespace
            json_str = json_str.strip()
            
            # Parse the JSON
            refined_categories = json.loads(json_str)
            
            logger.info(f"Generated refined categorization system with {len(refined_categories)} main categories")
            return refined_categories
            
        except Exception as e:
            logger.error(f"Failed to generate refined categories: {str(e)}", exc_info=True)
            # Fallback to a basic categorization system
            logger.info("Using fallback categorization system")
            return self._generate_fallback_refined_categories()
            
    def _generate_fallback_refined_categories(self) -> Dict[str, Dict[str, List[str]]]:
        """Generate a fallback refined categorization system if LLM fails"""
        # Create a more detailed version of our basic categories
        fallback = {
            "fps": {
                "frame_rate": ["fps", "framerate", "frame_rate", "frames_per_second"],
                "frame_time": ["frametime", "frame_time", "frame_latency", "rendertime"],
                "frame_pacing": ["framepacing", "frame_pacing", "jank", "stutter", "stability"]
            },
            "memory": {
                "usage": ["memory", "mem_usage", "ram", "memory_usage"],
                "allocation": ["alloc", "allocation", "memory_alloc", "heap"],
                "garbage_collection": ["gc", "garbage", "gctime", "collection"]
            },
            "cpu": {
                "utilization": ["cpu", "cpu_usage", "processor", "utilization"],
                "threading": ["thread", "threading", "core", "multicore", "worker"],
                "scheduling": ["scheduler", "scheduling", "priority", "process"]
            },
            "gpu": {
                "rendering": ["gpu", "render", "rendering", "graphics", "draw"],
                "shaders": ["shader", "fragment", "vertex", "pixel", "compute"],
                "pipeline": ["pipeline", "graphics_pipe", "render_pipe", "drawcall"]
            },
            "network": {
                "latency": ["ping", "latency", "delay", "rtt", "roundtrip"],
                "bandwidth": ["bandwidth", "throughput", "speed", "datarate"],
                "stability": ["packet", "loss", "jitter", "connection", "disconnect"]
            },
            "loading": {
                "times": ["load", "loading", "loadtime", "load_time", "startup"],
                "resources": ["resource", "asset", "file", "streaming", "preload"],
                "caching": ["cache", "cached", "precache", "buffer"]
            },
            "input": {
                "response": ["input", "response", "lag", "delay", "latency"],
                "devices": ["touch", "controller", "keyboard", "mouse", "gamepad"],
                "events": ["click", "press", "swipe", "gesture", "button"]
            },
            "audio": {
                "playback": ["audio", "sound", "playback", "play", "audio_play"],
                "mixing": ["mix", "mixing", "audio_mix", "channel", "voice"],
                "performance": ["audio_perf", "sound_perf", "audio_time", "audio_cpu"]
            },
            "battery": {
                "consumption": ["battery", "power", "energy", "consumption", "drain"],
                "temperature": ["battery_temp", "bat_temp", "power_temp", "energy_temp"],
                "charging": ["charge", "charging", "battery_level", "power_level"]
            },
            "thermal": {
                "device": ["thermal", "temperature", "temp", "heat", "device_temp"],
                "throttling": ["throttle", "thermal_throttle", "throttling", "downclock"],
                "cooling": ["cooling", "fan", "heatsink", "thermal_control"]
            }
        }
        
        return fallback
        
    def _match_to_refined_categories(
        self,
        column_name: str,
        basic_category: str,
        refined_categories: Dict[str, Dict[str, List[str]]]
    ) -> Tuple[str, str]:
        """
        Match a column name to the refined category system
        
        Args:
            column_name: The column name to match
            basic_category: The basic category from simple matching
            refined_categories: The refined categorization system
            
        Returns:
            A tuple of (main category, subcategory)
        """
        column_lower = column_name.lower()
        
        # First try to match within the expected main category (if it exists in the refined system)
        if basic_category in refined_categories:
            for subcategory, keywords in refined_categories[basic_category].items():
                for keyword in keywords:
                    if keyword in column_lower:
                        return basic_category, subcategory
        
        # If no match in the expected category, try all categories
        for main_category, subcategories in refined_categories.items():
            for subcategory, keywords in subcategories.items():
                for keyword in keywords:
                    if keyword in column_lower:
                        return main_category, subcategory
        
        # If still no match, return the basic category with "general" subcategory
        return basic_category, "general"