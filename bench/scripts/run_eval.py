from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

# Ensure repo root on path when running as a script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from bench.agents import AGENT_REGISTRY
from bench.eval.runner import load_jsonl, dump_jsonl, run_benchmark
from bench.eval.scorer import score_predictions
from bench.eval.judge import run_llm_judge
from bench.eval.validator import validate_dataset
from bench.eval.logging_utils import setup_logger
from bench.llm import LLMConfig, LLMRouter
from bench.embeddings import EmbeddingConfig, EmbeddingRouter, LangChainEmbeddingRouter
from bench.domains import detect_domain_from_episodes


DEFAULT_LLM_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-20241022",
    "gemini": "gemini-1.5-pro",
    "google": "gemini-1.5-pro",
    "deepseek": "deepseek-v4-pro",
    "llama": "meta/llama-4-maverick-17b-128e-instruct",
    "nvidia": "meta/llama-4-maverick-17b-128e-instruct",
    "stub": "stub",
}

DEFAULT_EMBED_MODELS = {
    "openai": "text-embedding-3-small",
    "hf": "sentence-transformers/all-MiniLM-L6-v2",
}

_OBVIOUS_OPENAI_EMBED_PREFIXES = ("text-embedding-",)
_OBVIOUS_HF_EMBED_PREFIXES = (
    "sentence-transformers/",
    "BAAI/",
    "intfloat/",
    "thenlper/",
    "jinaai/",
    "mixedbread-ai/",
    "WhereIsAI/",
)


def _default_llm_model(provider: str) -> str:
    key = (provider or "stub").lower().strip()
    try:
        return DEFAULT_LLM_MODELS[key]
    except KeyError as exc:  # pragma: no cover - argparse restricts provider values
        raise ValueError(f"Unsupported LLM provider for default model resolution: {provider}") from exc


def _resolve_llm_model_arg(provider: str, model: str | None) -> str:
    return model or _default_llm_model(provider)


def _default_embed_model(provider: str) -> str:
    try:
        return DEFAULT_EMBED_MODELS[provider]
    except KeyError as exc:  # pragma: no cover - argparse restricts provider values
        raise ValueError(f"Unsupported embedding provider for default model resolution: {provider}") from exc


def _is_obviously_openai_embedding_model(model: str) -> bool:
    return model.startswith(_OBVIOUS_OPENAI_EMBED_PREFIXES)


def _is_obviously_hf_embedding_model(model: str) -> bool:
    return model.startswith(_OBVIOUS_HF_EMBED_PREFIXES)


def _resolve_embedding_model_arg(provider: str, model: str | None) -> str:
    if not model:
        return _default_embed_model(provider)
    if provider == "openai" and _is_obviously_hf_embedding_model(model):
        raise SystemExit(
            "Embedding configuration mismatch: --embed_provider openai was used with an obvious HuggingFace model "
            f"({model}). Either set --embed_model to an OpenAI-compatible embedding model such as "
            f"'{_default_embed_model('openai')}', or switch --embed_provider to hf."
        )
    if provider == "hf" and _is_obviously_openai_embedding_model(model):
        raise SystemExit(
            "Embedding configuration mismatch: --embed_provider hf was used with an obvious OpenAI embedding model "
            f"({model}). Either set --embed_model to a local HF model such as "
            f"'{_default_embed_model('hf')}', or switch --embed_provider to openai."
        )
    return model


def _sanitize_run_name_component(value: str | None, *, fallback: str) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        raw = fallback
    chars = []
    prev_sep = False
    for ch in raw:
        if ch.isalnum():
            chars.append(ch)
            prev_sep = False
        elif not prev_sep:
            chars.append("_")
            prev_sep = True
    cleaned = "".join(chars).strip("_")
    return cleaned or fallback


def _rate(numer: int, denom: int) -> float:
    return float(numer) / float(denom) if denom > 0 else 0.0


def _combine_judge_answer_with_rule_context(
    *,
    rule_scores: list[dict],
    judge_rows: list[dict],
    judge_utility_accuracy: float,
) -> dict:
    """Combine LLM-judge answer metrics with rule-based prompt-context exposure.

    The LLM judge evaluates the final answer. Retrieval-stage privacy is
    deterministic over the prompt-visible memory block, so it comes from
    scores.jsonl. This helper joins them per checkpoint and computes exact
    end-to-end leakage rates:
      e2e leak = answer leak OR context exposure.
    """
    rule_by_id = {str(r.get("checkpoint_id") or ""): r for r in rule_scores}

    privacy_rows = []
    safety_rows = []
    for jr in judge_rows or []:
        cid = str(jr.get("checkpoint_id") or "")
        rule = rule_by_id.get(cid) or {}
        judge = jr.get("judge") if isinstance(jr.get("judge"), dict) else {}
        qtype = str(jr.get("query_type") or rule.get("query_type") or "")

        if qtype == "privacy":
            ans = judge.get("privacy_leak")
            ctx = rule.get("privacy_context_leak")
            if ans is not None or ctx is not None:
                privacy_rows.append(
                    {
                        "answer": bool(ans is True),
                        "context": bool(ctx is True),
                        "e2e": bool(ans is True or ctx is True),
                    }
                )
        elif qtype == "safety":
            ans = judge.get("deletion_leak")
            ctx = rule.get("deletion_context_leak")
            if ans is not None or ctx is not None:
                safety_rows.append(
                    {
                        "answer": bool(ans is True),
                        "context": bool(ctx is True),
                        "e2e": bool(ans is True or ctx is True),
                    }
                )

    privacy_context_rate = _rate(sum(1 for r in privacy_rows if r["context"]), len(privacy_rows))
    deletion_context_rate = _rate(sum(1 for r in safety_rows if r["context"]), len(safety_rows))
    privacy_e2e_rate = _rate(sum(1 for r in privacy_rows if r["e2e"]), len(privacy_rows))
    deletion_e2e_rate = _rate(sum(1 for r in safety_rows if r["e2e"]), len(safety_rows))

    return {
        "n_privacy_context_scored": len(privacy_rows),
        "n_safety_context_scored": len(safety_rows),
        "privacy_context_leakage_rate": privacy_context_rate,
        "deletion_context_leakage_rate": deletion_context_rate,
        "privacy_e2e_leakage_rate": privacy_e2e_rate,
        "deletion_e2e_leakage_rate": deletion_e2e_rate,
        "compliance_utility_e2e_score": float(judge_utility_accuracy)
        * (1.0 - privacy_e2e_rate)
        * (1.0 - deletion_e2e_rate),
        "context_metric_source": "rule_based_prompt_context_scan",
    }


def main() -> None:
    # Optional config file (YAML/JSON) support: load defaults from config, then let CLI override.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML/JSON config file. Values become defaults; CLI flags override.",
    )
    pre_args, _ = pre.parse_known_args()

    cfg_defaults = {}
    if pre_args.config:
        from bench.utils.config import flatten_config, load_config

        loaded = load_config(pre_args.config)
        cfg_defaults = flatten_config(loaded.data)

    p = argparse.ArgumentParser(parents=[pre])
    p.add_argument("--data_dir", type=str, default="bench/data/medical")
    p.add_argument(
        "--agent",
        type=str,
        choices=sorted(AGENT_REGISTRY.keys()),
        default="rag_naive",
    )
    p.add_argument("--top_k", type=int, default=20)

    # Mem0 baseline (fact extraction + memory update)
    p.add_argument(
        "--mem0_backend",
        type=str,
        default="upstream",
        choices=["builtin", "upstream"],
        help="Mem0 implementation backend: builtin (self-contained) or upstream (vendored official Memory()).",
    )

    p.add_argument(
        "--mem0_message_window",
        type=int,
        default=10,
        help="Mem0: number of most recent dialogue messages used for fact extraction.",
    )
    p.add_argument(
        "--mem0_top_s",
        type=int,
        default=5,
        help="Mem0: number of similar existing memories retrieved per new fact for update.",
    )
    p.add_argument(
        "--mem0_max_facts",
        type=int,
        default=20,
        help="Mem0: cap on number of extracted facts per update step.",
    )

    # Retrieval / chunking (RAG baselines)
    p.add_argument(
        "--retrieval_backend",
        type=str,
        default="embedding",
        choices=["tfidf", "embedding"],
        help="Retriever for RAG baselines. tfidf is offline; embedding calls an embedding model (real RAG).",
    )
    p.add_argument(
        "--chunk_mode",
        type=str,
        default="turn",
        choices=["turn", "window", "chars"],
        help="How to chunk dialogue turns into retrievable documents.",
    )
    p.add_argument("--chunk_window_turns", type=int, default=5)
    p.add_argument("--chunk_max_chars", type=int, default=4000)
    p.add_argument(
        "--use_faiss",
        action="store_true",
        help="Use FAISS for embedding retrieval if installed (pip install faiss-cpu).",
    )

    # A-Mem baseline (agentic memory)
    p.add_argument(
        "--a_mem_metadata_mode",
        type=str,
        default="llm",
        choices=["heuristic", "llm"],
        help="A-Mem memory-formation metadata mode. 'llm' is closer to the paper but more expensive.",
    )
    p.add_argument("--a_mem_link_top_m", type=int, default=3)
    p.add_argument("--a_mem_link_score_threshold", type=float, default=0.15)
    p.add_argument("--a_mem_graph_expand_hops", type=int, default=1)
    p.add_argument("--a_mem_graph_expand_per_hit", type=int, default=2)
    p.add_argument("--a_mem_rerank_role_bonus", type=float, default=0.08)
    p.add_argument("--a_mem_rerank_entity_bonus", type=float, default=0.06)

    # REMEM baseline (episodic hybrid graph memory)
    p.add_argument(
        "--remem_variant",
        type=str,
        default="iterative",
        choices=["iterative", "single"],
        help="REMem variant: iterative (REMem-I) or single (REMem-S).",
    )
    p.add_argument("--remem_max_steps", type=int, default=5)
    p.add_argument("--remem_retrieval_top_k", type=int, default=10, help="ReMem retrieval breadth before graph expansion.")
    p.add_argument("--remem_linking_top_k", type=int, default=5, help="ReMem per-focus graph expansion cap.")
    p.add_argument("--remem_qa_top_k", type=int, default=None, help="Final number of evidence items sent to the answer model. Defaults to --top_k.")
    p.add_argument("--remem_synonymy_threshold", type=float, default=0.8)

    p.add_argument(
        "--embedding_impl",
        type=str,
        default="native",
        choices=["native", "langchain"],
        help="Embedding RAG implementation. native is built-in; langchain uses langchain-core/community + FAISS (optional deps).",
    )

    # Embeddings (only used when retrieval_backend=embedding)
    p.add_argument(
        "--embed_provider",
        type=str,
        default="hf",
        choices=["openai", "hf"],
        help="Embedding provider. openai uses /v1/embeddings; hf uses local transformers models.",
    )
    p.add_argument(
        "--embed_model",
        type=str,
        default=None,
        help="Embedding model name. If omitted, a provider-aware default is chosen (text-embedding-3-small for openai; sentence-transformers/all-MiniLM-L6-v2 for hf).",
    )
    p.add_argument("--embed_api_base", type=str, default=None)
    p.add_argument("--embed_api_key_env", type=str, default=None)
    p.add_argument("--embed_batch_size", type=int, default=16)
    p.add_argument("--embed_device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--embed_max_length", type=int, default=512, help="Maximum input length for native HF embedding tokenization.")

    # LLM backend
    p.add_argument(
        "--llm_provider",
        type=str,
        default="openai",
        choices=["stub", "openai", "anthropic", "gemini", "deepseek", "llama", "nvidia"],
        help="Which hosted API to call. Use stub for offline deterministic mode.",
    )
    p.add_argument("--llm_model", type=str, default=None, help="Provider model name")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max_output_tokens", type=int, default=4096)
    p.add_argument(
        "--reasoning_effort",
        type=str,
        default=None,
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Provider-agnostic reasoning-effort hint. OpenAI maps to reasoning.effort, Gemini maps to thinking controls, Anthropic maps to output_config.effort/thinking.",
    )
    p.add_argument(
        "--text_verbosity",
        type=str,
        default=None,
        choices=["low", "medium", "high"],
        help="Text verbosity hint for providers that support it (currently OpenAI).",
    )
    p.add_argument("--timeout_s", type=float, default=60.0)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--api_base", type=str, default=None)
    p.add_argument("--api_key_env", type=str, default=None)
    p.add_argument("--anthropic_version", type=str, default="2023-06-01")
    p.add_argument(
        "--merge_system_into_user",
        action="store_true",
        help="Merge system prompt into user prompt (more robust across providers).",
    )

    # Stub-only behavior
    p.add_argument(
        "--llm_mode",
        type=str,
        default=None,
        help="leaky|obedient (only affects provider=stub).",
    )

    p.add_argument("--query_prompt_path", type=str, default=None)
    p.add_argument(
        "--answer_protocol",
        type=str,
        default="standard",
        choices=["standard", "native"],
        help="Answer-time memory presentation protocol. standard keeps the shared raw-snippet head; native enables backend-specific memory rendering where supported.",
    )

    # Optional: LLM-as-a-judge (additional scoring; does NOT replace rule-based scorer)
    p.add_argument("--use_llm_judge", action="store_true", help="Run an LLM judge and write judge_scores.jsonl")
    p.add_argument(
        "--judge_provider",
        type=str,
        default=None,
        choices=["stub", "openai", "anthropic", "gemini", "deepseek", "llama", "nvidia"],
        help="Judge provider (defaults to --llm_provider)",
    )
    p.add_argument("--judge_model", type=str, default=None, help="Judge model name (defaults to --llm_model)")
    p.add_argument("--judge_temperature", type=float, default=0.0)
    p.add_argument("--judge_max_output_tokens", type=int, default=4096)
    p.add_argument(
        "--judge_reasoning_effort",
        type=str,
        default=None,
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Judge-only reasoning-effort hint. Defaults to --reasoning_effort when omitted.",
    )
    p.add_argument(
        "--judge_text_verbosity",
        type=str,
        default=None,
        choices=["low", "medium", "high"],
        help="Judge-only text verbosity hint. Defaults to --text_verbosity when omitted.",
    )
    p.add_argument("--judge_prompt_path", type=str, default=None)

    # Concurrency
    p.add_argument(
        "--episode_concurrency",
        type=int,
        default=4,
        help="Run different episodes in parallel during baseline inference (1=serial).",
    )
    p.add_argument(
        "--judge_concurrency",
        type=int,
        default=4,
        help="Run LLM-judge in parallel across checkpoints (1=serial).",
    )

    # Optional: gate inner metrics by action correctness
    p.add_argument(
        "--gate_by_action",
        action="store_true",
        help=(
            "When set, treat inner metrics as failures whenever action_correct/action_ok is False. "
            "(utility_correct->False; privacy_leak->True; deletion_leak->True)"
        ),
    )

    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--run_name", type=str, default=None)

    # Scoring-only mode
    p.add_argument(
        "--score_only",
        action="store_true",
        help=(
            "Skip running the agent and only score an existing predictions.jsonl under out_dir/run_name. "
            "Useful when a run was interrupted and you want to compute metrics on the partial predictions."
        ),
    )

    # Reliability / UX
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing run_name output directory (append new predictions; skip completed checkpoints).",
    )
    p.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable tqdm progress bar.",
    )

    p.add_argument(
        "--show_ingest_progress",
        action="store_true",
        help="Show a nested progress bar for turn ingestion (useful for debugging).",
    )

    # Logging
    p.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log_file", type=str, default=None, help="Optional log file path under out_dir/run_name.")

    # Apply config defaults (only for known argparse destinations)
    if cfg_defaults:
        known_dests = {a.dest for a in p._actions}
        p.set_defaults(**{k: v for k, v in cfg_defaults.items() if k in known_dests})

    args = p.parse_args()

    args.llm_model = _resolve_llm_model_arg(args.llm_provider, args.llm_model)
    args.embed_model = _resolve_embedding_model_arg(args.embed_provider, args.embed_model)

    if (args.resume or args.score_only) and not args.run_name:
        raise SystemExit("--resume/--score_only requires --run_name so we know which output folder to use.")

    episodes_path = os.path.join(args.data_dir, "episodes.jsonl")
    ckpts_path = os.path.join(args.data_dir, "checkpoints.jsonl")

    if not (os.path.exists(episodes_path) and os.path.exists(ckpts_path)):
        raise SystemExit(
            "Invalid --data_dir. Expected a domain directory containing episodes.jsonl and checkpoints.jsonl, "
            f"but got: {args.data_dir}. For example, use --data_dir bench/data/medical."
        )

    episodes = load_jsonl(episodes_path)
    checkpoints = load_jsonl(ckpts_path)
    dataset_domain = detect_domain_from_episodes(episodes)
    run_domain = _sanitize_run_name_component(dataset_domain, fallback="generic")

    # Logger + outputs. Auto-generated run names include the dataset domain so
    # outputs are easier to scan, e.g. long_context_deepseek_medical_20260610_150404.
    run_name = args.run_name or (
        f"{args.agent}_{args.llm_provider}_{run_domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    mem0_cache_dir = os.path.join(out_dir, "mem0_cache")
    mem0_upstream_state_dir = os.path.join(out_dir, "mem0_upstream_state")
    log_file = None
    if args.log_file:
        log_file = args.log_file
        if not os.path.isabs(log_file):
            log_file = os.path.join(out_dir, args.log_file)
    logger = setup_logger(level=args.log_level, log_file=log_file)
    logger.info("Detected dataset domain: %s", dataset_domain)

    errors, warnings = validate_dataset(episodes=episodes, checkpoints=checkpoints, strict=True)
    for w in warnings:
        logger.warning("DATA WARNING: %s", w)
    if errors:
        for e in errors:
            logger.error("DATA ERROR: %s", e)
        raise SystemExit("Dataset validation failed. See errors above.")

    pred_path = os.path.join(out_dir, "predictions.jsonl")

    preds = None

    if args.score_only:
        if not os.path.exists(pred_path):
            raise SystemExit(
                f"--score_only was set but predictions.jsonl was not found at: {pred_path}. "
                "Provide the correct --run_name (and --out_dir if needed)."
            )
        preds = load_jsonl(pred_path)
        if not preds:
            raise SystemExit(f"predictions.jsonl exists but is empty: {pred_path}")
        logger.info("SCORE-ONLY: loaded %d predictions from %s", len(preds), pred_path)
    else:
        # Create LLM router for answering
        llm_cfg = LLMConfig(
            provider=args.llm_provider,
            model=args.llm_model,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
            text_verbosity=args.text_verbosity,
            timeout_s=args.timeout_s,
            max_retries=args.max_retries,
            api_base=args.api_base,
            api_key_env=args.api_key_env,
            anthropic_version=args.anthropic_version,
            merge_system_into_user=args.merge_system_into_user,
        )
        llm_router = LLMRouter(llm_cfg)

        # Create embedding router (only if needed).
        # For embedding_impl=langchain, we still build a router-like adapter so
        # agents such as A-Mem/ReMem/Mem0 builtin do not silently degrade or fail.
        embed_router = None
        if (
            args.retrieval_backend == "embedding"
            and (args.agent in {"rag_naive", "rag_policy", "a_mem", "remem"} or (args.agent == "mem0" and args.mem0_backend == "builtin"))
        ):
            embed_cfg = EmbeddingConfig(
                provider=args.embed_provider,
                model=args.embed_model,
                api_base=args.embed_api_base,
                api_key_env=args.embed_api_key_env,
                timeout_s=args.timeout_s,
                max_retries=args.max_retries,
                device=args.embed_device,
                batch_size=args.embed_batch_size,
                max_length=args.embed_max_length,
            )
            if args.embedding_impl == "native":
                embed_router = EmbeddingRouter(embed_cfg)
            else:
                embed_router = LangChainEmbeddingRouter(embed_cfg)

        AgentCls = AGENT_REGISTRY[args.agent]

        # Instantiate agent with supported kwargs
        kwargs = {"llm_router": llm_router, "query_prompt_path": args.query_prompt_path}

        if args.agent == "long_context":
            kwargs["max_turns"] = max(300, args.top_k)
        else:
            kwargs["top_k"] = args.top_k

        if args.llm_mode is not None:
            kwargs["llm_mode"] = args.llm_mode

        # Retrieval config for RAG agents
        if args.agent in {"rag_naive", "rag_policy"}:
            kwargs.update(
                {
                    "retrieval_backend": args.retrieval_backend,
                    "chunk_mode": args.chunk_mode,
                    "chunk_window_turns": args.chunk_window_turns,
                    "chunk_max_chars": args.chunk_max_chars,
                    "use_faiss": bool(args.use_faiss),
                    "embedding_impl": args.embedding_impl,
                    "embed_provider": args.embed_provider,
                    "embed_model": args.embed_model,
                    "embed_api_base": args.embed_api_base,
                    "embed_api_key_env": args.embed_api_key_env,
                    "embed_device": args.embed_device,
                    "embed_batch_size": args.embed_batch_size,
                }
            )
            if embed_router is not None:
                kwargs["embed_router"] = embed_router

        if args.agent == "a_mem":
            kwargs["answer_protocol"] = args.answer_protocol
            kwargs.update(
                {
                    "retrieval_backend": args.retrieval_backend,
                    "use_faiss": bool(args.use_faiss),
                    "metadata_mode": args.a_mem_metadata_mode,
                    "link_top_m": args.a_mem_link_top_m,
                    "link_score_threshold": args.a_mem_link_score_threshold,
                    "graph_expand_hops": args.a_mem_graph_expand_hops,
                    "graph_expand_per_hit": args.a_mem_graph_expand_per_hit,
                    "rerank_role_bonus": args.a_mem_rerank_role_bonus,
                    "rerank_entity_bonus": args.a_mem_rerank_entity_bonus,
                }
            )
            if embed_router is not None:
                kwargs["embed_router"] = embed_router

        if args.agent == "remem":
            kwargs["answer_protocol"] = args.answer_protocol
            kwargs.update(
                {
                    "variant": args.remem_variant,
                    "max_steps": args.remem_max_steps,
                    "retrieval_top_k": args.remem_retrieval_top_k,
                    "linking_top_k": args.remem_linking_top_k,
                    "qa_top_k": args.remem_qa_top_k,
                    "synonymy_threshold": args.remem_synonymy_threshold,
                }
            )
            if embed_router is not None:
                kwargs["embed_router"] = embed_router

        if args.agent == "mem0":
            # Mem0 backend: builtin (self-contained) or upstream (vendored official Memory())
            kwargs.update(
                {
                    "mem0_backend": args.mem0_backend,
                    "mem0_cache_dir": mem0_cache_dir,
                    "mem0_upstream_state_dir": mem0_upstream_state_dir,
                    "mem0_message_window": args.mem0_message_window,
                    "mem0_top_s": args.mem0_top_s,
                    "mem0_max_facts": args.mem0_max_facts,
                    "mem0_upstream_llm_provider": args.llm_provider,
                    "mem0_upstream_llm_model": args.llm_model,
                    "mem0_upstream_openai_base": args.api_base,
                    "mem0_upstream_api_key_env": args.api_key_env,
                    "mem0_upstream_embed_provider": args.embed_provider,
                    "mem0_upstream_embed_model": args.embed_model,
                    "mem0_upstream_embed_api_base": args.embed_api_base,
                    "mem0_upstream_embed_api_key_env": args.embed_api_key_env,
                    "mem0_upstream_embed_device": args.embed_device,
                }
            )

            if args.mem0_backend == "builtin":
                if embed_router is None:
                    raise ValueError(
                        "mem0 builtin backend requires an embedding router. "
                        "Please set --retrieval_backend embedding and configure a compatible embedding provider/model."
                    )
                kwargs["embed_router"] = embed_router

        agent = AgentCls(**kwargs)

        if (not args.resume) and os.path.exists(pred_path):
            # If the user reuses run_name without resume, start fresh.
            os.remove(pred_path)

        # Stream predictions to disk for crash-safe resume.
        # If episode_concurrency>1, we construct a fresh agent per episode.
        if int(args.episode_concurrency) > 1:
            def _agent_factory():
                return AgentCls(**kwargs)

            run_benchmark(
                agent_factory=_agent_factory,
                episodes=episodes,
                checkpoints=checkpoints,
                prediction_path=pred_path,
                resume=args.resume,
                show_progress=not args.no_progress,
                logger=logger,
                show_ingest_progress=args.show_ingest_progress,
                episode_concurrency=int(args.episode_concurrency),
            )
        else:
            run_benchmark(
                agent=agent,
                episodes=episodes,
                checkpoints=checkpoints,
                prediction_path=pred_path,
                resume=args.resume,
                show_progress=not args.no_progress,
                logger=logger,
                show_ingest_progress=args.show_ingest_progress,
                episode_concurrency=1,
            )

        preds = load_jsonl(pred_path)

    scores, rule_summary = score_predictions(
        episodes=episodes,
        checkpoints=checkpoints,
        predictions=preds,
        gate_by_action=bool(args.gate_by_action),
    )
    dump_jsonl(os.path.join(out_dir, "scores.jsonl"), scores)

    summary = dict(rule_summary)
    summary["rule_based"] = dict(rule_summary)

    # Optional LLM judge (primary metrics in v2)
    if args.use_llm_judge:
        judge_provider = args.judge_provider or args.llm_provider
        judge_model = args.judge_model or args.llm_model
        judge_prompt_path = args.judge_prompt_path or os.path.join(REPO_ROOT, 'bench', 'prompts', 'judge_prompt.txt')

        judge_cfg = LLMConfig(
            provider=judge_provider,
            model=judge_model,
            temperature=args.judge_temperature,
            max_output_tokens=args.judge_max_output_tokens,
            reasoning_effort=args.judge_reasoning_effort or args.reasoning_effort,
            text_verbosity=args.judge_text_verbosity or args.text_verbosity,
            timeout_s=args.timeout_s,
            max_retries=args.max_retries,
            api_base=args.api_base,
            api_key_env=args.api_key_env,
            anthropic_version=args.anthropic_version,
            merge_system_into_user=True,
        )
        judge_router = LLMRouter(judge_cfg)

        judge_out_path = os.path.join(out_dir, 'judge_scores.jsonl')
        if (not args.resume) and os.path.exists(judge_out_path):
            os.remove(judge_out_path)

        _, judge_summary = run_llm_judge(
            episodes=episodes,
            checkpoints=checkpoints,
            predictions=preds,
            judge_router=judge_router,
            prompt_path=judge_prompt_path,
            out_path=judge_out_path,
            resume=args.resume,
            gate_by_action=bool(args.gate_by_action),
            concurrency=int(args.judge_concurrency),
            logger=logger,
        )
        summary['llm_judge'] = judge_summary

        # Promote judge metrics to top-level for convenience (LLM-judge is primary).

        action_acc = float(judge_summary.get("judge_action_ok_rate") or 0.0)
        utility_acc = float(judge_summary.get("judge_effective_utility_accuracy") or 0.0)
        privacy_leak = float(judge_summary.get("judge_privacy_leakage_rate") or 0.0)
        deletion_leak = float(judge_summary.get("judge_deletion_leakage_rate") or 0.0)

        summary.update(
            {
                "n_checkpoints": int(judge_summary.get("n_judged") or rule_summary.get("n_checkpoints") or 0),
                "n_utility": int(judge_summary.get("n_utility") or rule_summary.get("n_utility") or 0),
                "action_accuracy": action_acc,
                "utility_accuracy": utility_acc,
                # Backward-compatible top-level leakage rates remain answer-level.
                "privacy_leakage_rate": privacy_leak,
                "deletion_leakage_rate": deletion_leak,
                "privacy_answer_leakage_rate": privacy_leak,
                "deletion_answer_leakage_rate": deletion_leak,
                # over_refusal is a behavioral metric best computed from predictions (rule-based summary)
                "over_refusal_rate": float(rule_summary.get("over_refusal_rate") or 0.0),
                "compliance_utility_score": utility_acc * (1.0 - privacy_leak) * (1.0 - deletion_leak),
            }
        )

        judge_rows = load_jsonl(judge_out_path, ignore_errors=True)
        context_aware_summary = _combine_judge_answer_with_rule_context(
            rule_scores=scores,
            judge_rows=judge_rows,
            judge_utility_accuracy=utility_acc,
        )
        summary.update(context_aware_summary)
        summary["llm_judge"]["context_aware"] = context_aware_summary


    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
