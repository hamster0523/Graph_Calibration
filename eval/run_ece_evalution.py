from pathlib import Path
import sys
import argparse
from typing import List, Dict, Any, Tuple, Optional
import uuid
import os
import numpy as np
import json
import torch
os.environ['HF_ENDPOINT'] = ""
os.environ['HF_HOME'] = ''
os.environ['HF_HUB_CACHE'] = ""
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
from transformers import AutoTokenizer
from tqdm import tqdm
from verl_hamster.Hamster_Generation_Manager import Hamster_LLM_Generation_Manager
from verl_hamster.Hamster_Generation_Manager import Hamster_Generation_Config, Hamster_Graph_Config
from verl_hamster.Hamster_Generation_Manager import Wrapped_VLLM_Client, Wrapped_VLLM_Config
from verl_hamster.verl.utils.hamster_utils.ece import compute
from verl_hamster.verl import DataProto
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from CPT_FactorGraph_Run.prompt import prompt_no_confidence, prompt_with_confidence, prompt_confidence_guided, calibration_prompt, calibration_prompt_with_example

def _infer_tp_size(default_tp: int = 1) -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        try:
            return max(1, len([x for x in cvd.split(",") if x.strip() != ""]))
        except Exception:
            return default_tp
    try:
        import subprocess
        out = subprocess.check_output(["nvidia-smi", "-L"]).decode("utf-8").strip()
        gpus = [l for l in out.splitlines() if l.strip().startswith("GPU ")]
        return max(1, len(gpus))
    except Exception:
        return default_tp

def _to_serializable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist() if x.numel() != 1 else x.detach().cpu().item()
    return x

def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize(v) for v in obj]
    return _to_serializable(obj)


def _load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _load_processed_batch_keys(summary_path: Path) -> Dict[str, Dict[str, Any]]:
    processed = {}
    for record in _load_jsonl_records(summary_path):
        batch_key = record.get("batch_key")
        if batch_key is not None:
            processed[batch_key] = record
    return processed

def in_debug_mode() -> bool:
    return sys.gettrace() is not None

def prepare():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--inference_data_dir", type=str, default="")
    argparser.add_argument("--model_name", type=str, default="")
    argparser.add_argument("--output_dir", type=str, default="")
    argparser.add_argument("--batch_size", type=int, default=8)
    argparser.add_argument("--max_new_tokens", type=int, default=512)
    argparser.add_argument("--temperature", type=float, default=0.8)
    argparser.add_argument("--top_p", type=float, default=0.9)
    argparser.add_argument("--is_verbal", action="store_true", help="Use verbal prompt with <confidence> tag")
    argparser.add_argument("--retriever_url", type=str, default="http://10.32.208.152:8100/retrieve", help="URL of the retriever server")
    argparser.add_argument("--use_local_calibrate", action="store_true", help="Whether to use local calibrate server")
    argparser.add_argument("--cpt_rpc_url", type=str, default="http://10.32.208.152:8101/calibrate")
    argparser.add_argument("--gpu_memory_utilization", type=float, default=0.7, help="GPU memory utilization for vLLM")

    args = argparser.parse_args()
        
    # load tokenizer   
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # initialize vLLM client
    tp = _infer_tp_size(default_tp=4) if not in_debug_mode() else 1
    print(f"Using tensor parallel size: {tp}")
    vllm_client = LLM(
        model=args.model_name,
        tensor_parallel_size=tp,
        pipeline_parallel_size=1,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False if not in_debug_mode() else True,
    )
    
    if args.use_local_calibrate:
        graph_config = Hamster_Graph_Config(
                cpt_model_path=""
                )
    else:
        assert args.cpt_rpc_url is not None, "cpt_rpc_url must be provided if not using local calibrate"
        graph_config = Hamster_Graph_Config(
                cpt_rpc_url=args.cpt_rpc_url,
                )
    
    generation_config = Hamster_Generation_Config(
        max_turns=5,
        max_start_length=2048,
        max_prompt_length=2048,
        max_response_length=512,
        max_obs_length=500,
        num_gpus=4,
        search_url=args.retriever_url,
        is_verbal=args.is_verbal,
        graph_config=graph_config,
    )
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=-1,
        n=1,
        max_tokens=args.max_new_tokens,
        logprobs=1,
    )
    
    vllm_config = Wrapped_VLLM_Config(sampling_params = sampling_params)

    generation_manager = Hamster_LLM_Generation_Manager(
        tokenizer = tokenizer,
        actor_rollout_wg = None,
        config = generation_config,
        test_vllm_engine = Wrapped_VLLM_Client(llm=vllm_client, config=vllm_config)
    )
    
    return {
        "args": args,
        "generation_manager": generation_manager,
        "is_verbal": args.is_verbal
    }
    
def format_batch_data(
    batch_data: List[Dict[str, Any]],
    generation_manager: Hamster_LLM_Generation_Manager,
    *,
    do_sample: bool = False,
    validate: bool = False,
    is_verbal: bool = False
) -> DataProto:
    if not batch_data:
        raise ValueError("batch_data must contain at least one sample")
    if generation_manager is None:
        raise ValueError("generation_manager is required to format batch data")

    tokenizer = generation_manager.tokenizer
    config = generation_manager.config

    questions = [example.get("question", "") for example in batch_data]
    sample_ids = [example.get("id", str(uuid.uuid4())) for example in batch_data]

    if not is_verbal:
        prompts = [calibration_prompt_with_example.format(question=question) for question in questions]
    else:
        prompts = [calibration_prompt_with_example.format(question=question) for question in questions]
        
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_prompt_length,
    )

    input_ids = encoded["input_ids"].long()
    attention_mask = encoded["attention_mask"].long()
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids = position_ids.clamp_min(0)

    tensor_batch = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=[input_ids.size(0)],
    )

    golden_answers = []
    for example in batch_data:
        if "expected_answer" in example and example["expected_answer"] is not None:
            golden_answers.append(example["expected_answer"])
        elif example.get("golden_answers"):
            golden_answers.append(example["golden_answers"])
        else:
            golden_answers.append(example.get("answer"))

    non_tensor_batch = {
        "golden_answers": np.array(golden_answers, dtype=object),
        "prompts": np.array(prompts, dtype=object),
        "sample_id": np.array(sample_ids, dtype=object),
    }

    meta_info = {
        "eos_token_id": [tokenizer.eos_token_id],
        "do_sample": do_sample,
        "validate": validate,
        "questions": questions,
        "question": questions,
        "prompts": prompts,
        "sample_id": sample_ids,
    }

    return DataProto(batch=tensor_batch, meta_info=meta_info, non_tensor_batch=non_tensor_batch)

def post_process_output_dataproto(output_proto: DataProto):
    compute_batch = compute(output_proto)
    batch_size = compute_batch.batch.batch_size[0]
    meta_info = compute_batch.meta_info or {}

    mean_forward_list = meta_info.get("mean_forward", [])
    rms_forward_list = meta_info.get("rms_forward", [])
    gmean_forward_list = meta_info.get("gmean_forward", [])
    max_forward_list = meta_info.get("max_forward", [])
    kl_weighted_forward_list = meta_info.get("kl_weighted_forward", [])

    mean_backward_list = meta_info.get("mean_backward", [])
    rms_backward_list = meta_info.get("rms_backward", [])
    gmean_backward_list = meta_info.get("gmean_backward", [])
    max_backward_list = meta_info.get("max_backward", [])
    kl_weighted_backward_list = meta_info.get("kl_weighted_backward", [])

    mean_verbal_list = meta_info.get("mean_verbal", [])
    rms_verbal_list = meta_info.get("rms_verbal", [])
    gmean_verbal_list = meta_info.get("gmean_verbal", [])
    max_verbal_list = meta_info.get("max_verbal", [])

    return_batch_info_dict = {
        "batch_accumulate_mean_ece_forward": meta_info.get("mean_ece_forward"),
        "batch_accumulate_rms_ece_forward": meta_info.get("rms_ece_forward"),
        "batch_accumulate_gmean_ece_forward": meta_info.get("gmean_ece_forward"),
        "batch_accumulate_max_ece_forward": meta_info.get("max_ece_forward"),
        "batch_accumulate_kl_weighted_ece_forward": meta_info.get("kl_weighted_ece_forward"),

        "batch_accumulate_mean_ece_backward": meta_info.get("mean_ece_backward"),
        "batch_accumulate_rms_ece_backward": meta_info.get("rms_ece_backward"),
        "batch_accumulate_gmean_ece_backward": meta_info.get("gmean_ece_backward"),
        "batch_accumulate_max_ece_backward": meta_info.get("max_ece_backward"),
        "batch_accumulate_kl_weighted_ece_backward": meta_info.get("kl_weighted_ece_backward"),

        "batch_accumulate_mean_auc_forward": meta_info.get("mean_auc_forward"),
        "batch_accumulate_rms_auc_forward": meta_info.get("rms_auc_forward"),
        "batch_accumulate_gmean_auc_forward": meta_info.get("gmean_auc_forward"),
        "batch_accumulate_max_auc_forward": meta_info.get("max_auc_forward"),
        "batch_accumulate_kl_weighted_auc_forward": meta_info.get("kl_weighted_auc_forward"),

        "batch_accumulate_mean_auc_backward": meta_info.get("mean_auc_backward"),
        "batch_accumulate_rms_auc_backward": meta_info.get("rms_auc_backward"),
        "batch_accumulate_gmean_auc_backward": meta_info.get("gmean_auc_backward"),
        "batch_accumulate_max_auc_backward": meta_info.get("max_auc_backward"),
        "batch_accumulate_kl_weighted_auc_backward": meta_info.get("kl_weighted_auc_backward"),
        "acc": meta_info.get("acc", 0.0),
    }

    return_batch_info_dict.update({
        "batch_accumulate_mean_ece_verbal": meta_info.get("mean_ece_verbal"),
        "batch_accumulate_rms_ece_verbal": meta_info.get("rms_ece_verbal"),
        "batch_accumulate_gmean_ece_verbal": meta_info.get("gmean_ece_verbal"),
        "batch_accumulate_max_ece_verbal": meta_info.get("max_ece_verbal"),

        "batch_accumulate_mean_auc_verbal": meta_info.get("mean_auc_verbal"),
        "batch_accumulate_rms_auc_verbal": meta_info.get("rms_auc_verbal"),
        "batch_accumulate_gmean_auc_verbal": meta_info.get("gmean_auc_verbal"),
        "batch_accumulate_max_auc_verbal": meta_info.get("max_auc_verbal"),
    })

    acc = meta_info.get("acc", 0.0)
    graphs = compute_batch.non_tensor_batch.get("graph_calibrations", [])
    if len(graphs) != batch_size:
        raise ValueError(f"graph_calibrations length {len(graphs)} != batch_size {batch_size}")

    def _safe_get(lst, i, default=float('nan')):
        return lst[i] if i < len(lst) else default

    return_list_of_dict = []
    for i in range(batch_size):
        d_src = graphs[i] if isinstance(graphs[i], dict) else {}
        d = dict(d_src)  

        d['mean_forward'] = _safe_get(mean_forward_list, i)
        d['rms_forward'] = _safe_get(rms_forward_list, i)
        d['gmean_forward'] = _safe_get(gmean_forward_list, i)
        d['max_forward'] = _safe_get(max_forward_list, i)
        d['kl_weighted_forward'] = _safe_get(kl_weighted_forward_list, i)

        d['mean_backward'] = _safe_get(mean_backward_list, i)
        d['rms_backward'] = _safe_get(rms_backward_list, i)
        d['gmean_backward'] = _safe_get(gmean_backward_list, i)
        d['max_backward'] = _safe_get(max_backward_list, i)
        d['kl_weighted_backward'] = _safe_get(kl_weighted_backward_list, i)

        d['mean_verbal'] = _safe_get(mean_verbal_list, i)
        d['rms_verbal'] = _safe_get(rms_verbal_list, i)
        d['gmean_verbal'] = _safe_get(gmean_verbal_list, i)
        d['max_verbal'] = _safe_get(max_verbal_list, i)

        d['batch_acc'] = acc
        d['acc'] = acc
        d['batch_size'] = batch_size

        return_list_of_dict.append(d)

    return return_list_of_dict, return_batch_info_dict

def run_evaluation(args, generation_manager: Hamster_LLM_Generation_Manager, inference_files, is_verbal: bool = False):

    import re
    import hashlib
    from pathlib import Path
    import json
    import numpy as np
    from tqdm import tqdm
    import uuid

    def _safe_tag(s: str) -> str:
        base = Path(s).name
        return re.sub(r'[^0-9A-Za-z._-]+', '_', base)

    def _stable_sample_id_from_question(question: Optional[str]) -> str:
        if not question:
            return ""
        q_norm = " ".join(question.strip().split())
        return hashlib.sha1(q_norm.encode("utf-8")).hexdigest()

    def _get_sample_id_from_example(example: Dict[str, Any]) -> str:
        sid = example.get("id")
        if sid is not None and str(sid) != "":
            return str(sid)
        return _stable_sample_id_from_question(example.get("question", ""))

    def _split_meta_per_sample(meta: Dict[str, Any], batch_size: int) -> List[Dict[str, Any]]:
        per_sample = [dict() for _ in range(batch_size)]
        for k, v in (meta or {}).items():
            if isinstance(v, (list, tuple)) and len(v) == batch_size:
                for i in range(batch_size):
                    per_sample[i][k] = v[i]
            else:
                for i in range(batch_size):
                    per_sample[i][k] = v
        return per_sample


    results_root = Path(args.output_dir).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    summary_file_path = results_root / "ece_summary.txt"

    for inference_file in inference_files:
        print(f"-------------- Processing file: {inference_file} --------------")

        inference_file_path = Path(args.inference_data_dir) / inference_file
        model_tag = _safe_tag(args.model_name)
        data_tag = _safe_tag(Path(inference_file).name)

        output_file_path = results_root / f"{model_tag}__ece_{data_tag}"
        batch_summary_path = results_root / f"{model_tag}__batch_summary_{data_tag}"
        per_sample_meta_path = results_root / f"{model_tag}__sample_meta_{data_tag}"

        print("Writing sample results to:", output_file_path.resolve())
        print("Writing batch summaries to:", batch_summary_path.resolve())
        print("Writing per-sample meta to:", per_sample_meta_path.resolve())

        with open(inference_file_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        batch_size = args.batch_size

        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        output_file_path.touch(exist_ok=True)
        batch_summary_path.touch(exist_ok=True)
        per_sample_meta_path.touch(exist_ok=True)

        existing_records = _load_jsonl_records(output_file_path)
        processed_sample_ids = set()
        for rec in existing_records:
            sid = rec.get("sample_id")
            if sid:
                processed_sample_ids.add(str(sid))
                continue
            q = rec.get("question")
            if q:
                processed_sample_ids.add(_stable_sample_id_from_question(q))

        print(f"Found {len(processed_sample_ids)} previously processed samples in existing results.")

        processed_batches_current_run = 0
        skipped_batches = 0

        for batch_index, start_idx in enumerate(
            tqdm(range(0, len(data), batch_size), desc=f"Processing {inference_file}")
        ):
            end_idx = min(start_idx + batch_size, len(data))
            batch_data = data[start_idx:end_idx]

            stable_ids_list = [_get_sample_id_from_example(ex) for ex in batch_data]
            if stable_ids_list and all(sid and sid in processed_sample_ids for sid in stable_ids_list):
                skipped_batches += 1
                continue

            data_proto = format_batch_data(
                batch_data,
                generation_manager,
                do_sample=True,
                validate=False,
                is_verbal=is_verbal
            )
            initial_input_ids = data_proto.batch["input_ids"][
                :, -generation_manager.config.max_start_length:
            ].clone()
            output_proto = generation_manager.run_llm_loop(
                data_proto, initial_input_ids=initial_input_ids
            )

            return_list_of_dict, return_batch_info_dict = post_process_output_dataproto(output_proto)

            output_proto_meta_info_dict: Dict = output_proto.meta_info or {}

            try:
                per_sample_metas = _split_meta_per_sample(
                    output_proto_meta_info_dict, len(batch_data)
                )
                sample_ids_for_meta = [_get_sample_id_from_example(ex) for ex in batch_data]

                with open(per_sample_meta_path, "a", encoding="utf-8") as f_ps:
                    for i, sm_meta in enumerate(per_sample_metas):
                        rec = {
                            "dataset": inference_file,
                            "model": args.model_name,
                            "batch_key": f"{data_tag}:{batch_index}",
                            "batch_index": batch_index,
                            "index_in_batch": i,
                            "global_index": start_idx + i,
                            "batch_size": len(batch_data),
                            "sample_id": sample_ids_for_meta[i],
                            "meta_info": sanitize(sm_meta),
                        }
                        f_ps.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[WARN] Failed to write per-sample meta_info: {e}")

            sample_ids_written = []
            with open(output_file_path, "a", encoding="utf-8") as f_out:
                for ex, result in zip(batch_data, return_list_of_dict):
                    sid = _get_sample_id_from_example(ex)
                    if not sid:
                        sid = _stable_sample_id_from_question(result.get("question", "")) or str(uuid.uuid4())

                    if sid in processed_sample_ids:
                        continue  

                    sanitized = sanitize(result)
                    sanitized["sample_id"] = sid  
                    f_out.write(json.dumps(sanitized, ensure_ascii=False) + "\n")
                    processed_sample_ids.add(sid)
                    sample_ids_written.append(sid)

            summary_entry = {
                "dataset": inference_file,
                "model": args.model_name,
                "batch_key": f"{data_tag}:{batch_index}",
                "batch_index": batch_index,
                "start_index": start_idx,
                "end_index": end_idx - 1,
                "batch_size": len(batch_data),
                "sample_ids": sample_ids_written,  
                "metrics": sanitize(return_batch_info_dict),
            }
            with open(batch_summary_path, "a", encoding="utf-8") as f_summary_batch:
                f_summary_batch.write(json.dumps(summary_entry, ensure_ascii=False) + "\n")

            processed_batches_current_run += 1

        print(f"Results saved to {output_file_path.resolve()}")
        print(f"Processed batches this run: {processed_batches_current_run}, skipped: {skipped_batches}")

        all_batch_records = _load_jsonl_records(batch_summary_path)
        all_batch_records = [r for r in all_batch_records if r.get("dataset") == inference_file]

        total_batches_recorded = len(all_batch_records)
        total_examples_recorded = 0
        aggregated_totals: Dict[str, float] = {}
        for record in all_batch_records:
            total_examples_recorded += record.get("batch_size", 0)
            metrics = record.get("metrics", {})
            for key, value in metrics.items():
                try:
                    value_float = float(value)
                except (TypeError, ValueError):
                    continue
                aggregated_totals[key] = aggregated_totals.get(key, 0.0) + value_float

        divisor = total_batches_recorded if total_batches_recorded else 1
        summary_lines = [
            f"dataset: {inference_file}",
            f"model: {args.model_name}",
            f"total_examples: {total_examples_recorded}",
            f"batches: {total_batches_recorded}",
        ]
        for key in sorted(aggregated_totals.keys()):
            metric_name = key.replace("batch_accumulate_", "") + "_avg"
            summary_lines.append(f"{metric_name}: {aggregated_totals[key] / divisor:.6f}")
        summary_lines.append("")

        with open(summary_file_path, "a", encoding="utf-8") as f_summary:
            f_summary.write("\n".join(summary_lines))

def main():
    pre_dict = prepare()
    args = pre_dict["args"]
    generation_manager = pre_dict["generation_manager"]
    is_verbal = pre_dict["is_verbal"]

    # find all jsonl files in inference_data_dir
    inference_files = [f for f in os.listdir(args.inference_data_dir) if f.endswith(".jsonl")]
    print(f"Found {len(inference_files)} inference files.")
    
    run_evaluation(
        args,
        generation_manager,
        inference_files,
        is_verbal=is_verbal
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Process interrupted by user. Exiting...")
        sys.exit(0)