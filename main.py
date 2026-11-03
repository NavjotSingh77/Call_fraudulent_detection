"""
Main CLI entry point for the Fraud Call Detection system.

Commands:
    download-data     Download NCSU dataset (metadata + audio)
    validate-data     Validate downloaded audio files and manifest
    preprocess        Preprocess all audio files (resample, mono, normalize)
    transcribe        Transcribe audio files with Whisper
    prepare-dataset   Build dataset manifest and train/val/test splits
    train             Train the BiLSTM model
    evaluate          Evaluate trained model on test or validation set
    predict           Predict fraud likelihood on a single audio file
    serve             Launch FastAPI backend server

Usage:
    python main.py download-data [--max-files N] [--force]
    python main.py validate-data
    python main.py preprocess
    python main.py transcribe [--model tiny|base|small|medium] [--force]
    python main.py prepare-dataset
    python main.py train
    python main.py evaluate [--split test]
    python main.py predict --audio path/to/call.wav [--segments]
    python main.py serve [--host HOST] [--port PORT]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.config import Config, load_config, resolve_path
from src.utils.logger import get_logger, setup_logger

logger = logging.getLogger("fraud_detection.cli")


def _init_logging(config: Dict[str, Any]) -> None:
    """Initialize structured logger based on configuration."""
    log_cfg = config.get("logging", {})
    level = log_cfg.get("level", "INFO")
    log_file = log_cfg.get("file")
    log_file_path = (_PROJECT_ROOT / log_file) if log_file else None
    setup_logger(name="fraud_detection", level=level, log_file=log_file_path)


# ---------------------------------------------------------------------------
# Command Handlers
# ---------------------------------------------------------------------------


def cmd_download_data(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    """Download NCSU metadata and audio files."""
    from src.data.ingestion import run_ingestion

    print("=" * 60)
    print("NCSU ROBOCALL AUDIO DATASET INGESTION")
    print("=" * 60)

    # Ensure project_root is present in config dict
    cfg = dict(config)
    cfg["project_root"] = str(_PROJECT_ROOT)

    max_files = getattr(args, "max_files", None)
    force = getattr(args, "force", False)

    try:
        manifest_df = run_ingestion(cfg, max_files=max_files, force=force)
        print(f"\n[SUCCESS] Ingestion completed. Manifest has {len(manifest_df)} records.")
        print(f"Manifest saved to: data/metadata/manifest.csv")
        return 0
    except Exception as exc:
        logger.exception("Error during dataset download: %s", exc)
        print(f"\n[ERROR] Dataset download failed: {exc}")
        return 1


def cmd_validate_data(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    """Validate audio files, transcripts, and manifest integrity."""
    from src.data.dataset_validator import generate_validation_report, validate_manifest
    from src.data.ingestion import load_manifest

    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    cfg = dict(config)
    cfg["project_root"] = str(_PROJECT_ROOT)

    try:
        manifest_df = load_manifest(cfg)
    except FileNotFoundError:
        print("[ERROR] Manifest not found. Please run 'python main.py download-data' first.")
        return 1

    print(f"Loaded manifest with {len(manifest_df)} rows. Validating audio & metadata...")
    validated_df = validate_manifest(manifest_df, cfg)

    report_path = resolve_path(config, "paths.reports") / "validation_report.json"
    summary = generate_validation_report(validated_df, report_path)

    print("\n--- Validation Summary ---")
    print(f"Total records:      {summary.get('total_records', 0)}")
    print(f"Audio checked:      {summary.get('audio_files_checked', 0)}")
    print(f"Audio valid:        {summary.get('audio_valid_count', 0)}")
    print(f"Audio corrupt/inv:  {summary.get('audio_invalid_count', 0)}")
    print(f"Transcripts pres:   {summary.get('transcripts_present', 0)}")
    print(f"Report saved to:    {report_path}")

    return 0


def cmd_preprocess(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    """Preprocess audio files: mono conversion, 16kHz resampling, amplitude normalization."""
    from src.audio.preprocessing import preprocess_batch
    from src.data.ingestion import load_manifest

    print("=" * 60)
    print("AUDIO PREPROCESSING (16kHz Mono Peak-Normalized)")
    print("=" * 60)

    cfg = dict(config)
    cfg["project_root"] = str(_PROJECT_ROOT)

    try:
        manifest_df = load_manifest(cfg)
    except FileNotFoundError:
        print("[ERROR] Manifest not found. Please run 'python main.py download-data' first.")
        return 1

    audio_files = []
    if "audio_path" in manifest_df.columns:
        for p in manifest_df["audio_path"].dropna():
            path_obj = Path(p)
            if not path_obj.is_absolute():
                path_obj = _PROJECT_ROOT / path_obj
            if path_obj.exists():
                audio_files.append(path_obj)

    if not audio_files:
        print("[WARNING] No local audio files found in manifest to preprocess.")
        print("Run 'python main.py download-data' to fetch audio files.")
        return 0

    print(f"Found {len(audio_files)} audio files to preprocess.")
    out_dir = resolve_path(config, "paths.data_processed_audio")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = preprocess_batch(audio_files, out_dir, config)
    successes = sum(1 for r in results if r.get("processed", False) or r.get("success", False))
    print(f"\n[SUCCESS] Preprocessing completed. {successes}/{len(results)} processed successfully.")
    print(f"Processed audio stored in: {out_dir}")
    return 0


def cmd_transcribe(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    """Transcribe audio files using OpenAI Whisper."""
    from src.audio.transcription import (
        load_existing_transcript,
        save_transcript_cache,
        transcribe_audio,
    )
    from src.data.ingestion import load_manifest

    print("=" * 60)
    print("SPEECH-TO-TEXT TRANSCRIPTION (Whisper)")
    print("=" * 60)

    cfg = dict(config)
    cfg["project_root"] = str(_PROJECT_ROOT)

    if hasattr(args, "model") and args.model:
        cfg.setdefault("transcription", {})["whisper_model"] = args.model

    try:
        manifest_df = load_manifest(cfg)
    except FileNotFoundError:
        print("[ERROR] Manifest not found. Run 'python main.py download-data' first.")
        return 1

    transcripts_dir = resolve_path(config, "paths.data_processed_transcripts")
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    force = getattr(args, "force", False)
    processed_count = 0

    for _, row in manifest_df.iterrows():
        call_id = str(row["call_id"])
        audio_path_str = row.get("audio_path")
        existing_transcript = row.get("transcript")
        language = str(row.get("language", "en"))

        # If transcript already provided by NCSU and not forced, cache it directly
        if existing_transcript and pd.notna(existing_transcript) and not force:
            result = load_existing_transcript(
                transcript_text=str(existing_transcript),
                call_id=call_id,
                audio_path=str(audio_path_str) if pd.notna(audio_path_str) else "",
                language=language,
            )
            save_transcript_cache(result, transcripts_dir)
            processed_count += 1
            continue

        # If local audio exists, run Whisper transcription
        if audio_path_str and pd.notna(audio_path_str):
            a_path = Path(audio_path_str)
            if not a_path.is_absolute():
                a_path = _PROJECT_ROOT / a_path
            if a_path.exists():
                try:
                    result = transcribe_audio(
                        audio_path=a_path,
                        call_id=call_id,
                        config=cfg,
                        transcripts_dir=transcripts_dir,
                        force=force,
                    )
                    processed_count += 1
                except Exception as exc:
                    logger.warning("Transcription failed for %s: %s", call_id, exc)

    print(f"\n[SUCCESS] Transcription phase completed. Processed {processed_count} transcripts.")
    print(f"Transcripts cached in: {transcripts_dir}")
    return 0


def cmd_prepare_dataset(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    """Prepare dataset manifest and split into train/val/test with group-leakage prevention."""
    from src.data.dataset_loader import check_class_balance, filter_for_training
    from src.data.ingestion import load_manifest
    from src.data.splitting import create_splits, save_splits

    print("=" * 60)
    print("DATASET PREPARATION & GROUP-AWARE SPLITTING")
    print("=" * 60)

    cfg = dict(config)
    cfg["project_root"] = str(_PROJECT_ROOT)

    try:
        manifest_df = load_manifest(cfg)
    except FileNotFoundError:
        print("[ERROR] Manifest not found. Run 'python main.py download-data' first.")
        return 1

    eligible_df = filter_for_training(manifest_df)
    balance_info = check_class_balance(eligible_df)

    print(f"Eligible records for training: {len(eligible_df)}")
    print(f"Class distribution: {balance_info.get('class_counts')}")

    if balance_info.get("imbalance_ratio") == float("inf") or len(balance_info.get("class_counts", {})) < 2:
        print("\n[IMPORTANT RESEARCH NOTICE]:")
        print("The NCSU dataset contains solely confirmed robocall/fraud cases (FTC enforcement backed).")
        print("To train a binary classifier, legitimate non-fraud calls (e.g. from LibriSpeech or Common Voice)")
        print("must be added to data/external/ to establish the negative class.")
        print("The system strictly avoids fabricating pseudo-legitimate labels.")

    train_df, val_df, test_df = create_splits(
        df=eligible_df,
        config=config,
        random_seed=config.get("project", {}).get("random_seed", 42),
    )

    save_splits(train_df, val_df, test_df, config)
    print(f"\n[SUCCESS] Splits saved: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return 0


def cmd_train(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    """Train the BiLSTM classifier."""
    from src.training.train import InsufficientDataError, run_training

    print("=" * 60)
    print("MODEL TRAINING (BiLSTM Transcript Classifier)")
    print("=" * 60)

    try:
        history = run_training(config=config)
        print("\n[SUCCESS] Training completed successfully!")
        return 0
    except InsufficientDataError as ide:
        print(f"\n[CANNOT PROCEED WITH SUPERVISED TRAINING]")
        print(str(ide))
        print("\nScientific integrity: Fabricating negative samples was avoided.")
        return 1
    except Exception as exc:
        logger.exception("Training error: %s", exc)
        print(f"\n[ERROR] Training failed: {exc}")
        return 1


def cmd_evaluate(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    """Evaluate trained model on test or validation split."""
    from src.training.evaluate import run_evaluation

    split = getattr(args, "split", "test")
    print("=" * 60)
    print(f"MODEL EVALUATION (Split: {split})")
    print("=" * 60)

    try:
        metrics = run_evaluation(config=config, split=split)
        print(f"\nEvaluation Results on '{split}':")
        print(f"  Accuracy:  {metrics.get('accuracy', 0.0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0.0):.4f}")
        print(f"  Recall:    {metrics.get('recall', 0.0):.4f}")
        print(f"  F1 Score:  {metrics.get('f1', 0.0):.4f}")
        print(f"  ROC-AUC:   {metrics.get('roc_auc', 0.0):.4f}")
        print(f"  PR-AUC:    {metrics.get('pr_auc', 0.0):.4f}")
        return 0
    except Exception as exc:
        print(f"\n[ERROR] Evaluation failed: {exc}")
        return 1


def cmd_predict(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    """Run inference on a single audio file and output structured JSON report."""
    from src.audio.preprocessing import preprocess_audio, validate_audio_format
    from src.audio.segmentation import segment_audio
    from src.audio.transcription import transcribe_audio
    from src.features.text_preprocessing import extract_fraud_features
    from src.models.bilstm import load_model_artifacts
    from src.models.inference import (
        PredictionResult,
        aggregate_segment_predictions,
        get_confidence_level,
        get_prediction_label,
        predict_transcript,
        result_to_json,
    )

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"[ERROR] Audio file not found: {audio_path}")
        return 1

    call_id = audio_path.stem
    print(f"Processing audio: {audio_path} (call_id: {call_id})")

    # 1. Audio validation
    fmt_info = validate_audio_format(audio_path)
    if not fmt_info["valid"]:
        print(f"[ERROR] Invalid audio format: {fmt_info.get('error')}")
        return 1

    # 2. Audio preprocessing
    proc_dir = resolve_path(config, "paths.data_processed_audio")
    proc_info = preprocess_audio(audio_path, proc_dir, config)
    out_p = proc_info.get("processed_path") or proc_info.get("output_path")
    if not out_p:
        print(f"[ERROR] Audio preprocessing failed: {proc_info.get('error')}")
        return 1
    processed_audio_path = Path(out_p)

    # 3. Whisper transcription
    transcripts_dir = resolve_path(config, "paths.data_processed_transcripts")
    trans_result = transcribe_audio(
        audio_path=processed_audio_path,
        call_id=call_id,
        config=config,
        transcripts_dir=transcripts_dir,
    )

    transcript_text = trans_result.transcript
    language = trans_result.language

    # 4. Feature extraction
    fraud_feat = extract_fraud_features(transcript_text)

    # 5. Model prediction
    models_dir = resolve_path(config, "paths.models")
    try:
        model, vectorizer_config, label_encoder, model_config = load_model_artifacts(models_dir)
        pred_result = predict_transcript(
            transcript=transcript_text,
            call_id=call_id,
            language=language,
            model=model,
            vectorizer_config=vectorizer_config,
            label_encoder=label_encoder,
            config=config,
            fraud_features=fraud_feat,
        )
    except Exception as exc:
        # Fallback if model not trained yet: rule/feature-based heuristic estimation
        logger.warning("Trained model not available (%s); generating heuristic evaluation.", exc)
        score = 0.0
        if fraud_feat.contains_urgency_terms:
            score += 0.3
        if fraud_feat.contains_threat_terms:
            score += 0.3
        if fraud_feat.contains_impersonation_terms:
            score += 0.3
        if fraud_feat.contains_financial_terms:
            score += 0.2
        if fraud_feat.contains_otp_terms:
            score += 0.3
        prob = min(float(score), 0.99)
        conf = get_confidence_level(prob, config)
        overall = get_prediction_label(prob, config)
        review_req = conf != "high"

        from src.models.inference import extract_risk_indicators
        risk_ind = extract_risk_indicators(transcript_text, fraud_feat)

        pred_result = PredictionResult(
            call_id=call_id,
            language=language,
            overall_prediction=overall,
            fraud_probability=prob,
            confidence=conf,
            segments=[],
            risk_indicators=risk_ind,
            transcript=transcript_text,
            review_required=review_req,
            model_version="heuristic-baseline",
        )

    # 6. Segment-level analysis if requested
    if getattr(args, "segments", False):
        try:
            import soundfile as sf
            audio_data, sr = sf.read(str(processed_audio_path))
            seg_duration = config.get("audio", {}).get("segment_duration_seconds", 30)
            seg_stride = config.get("audio", {}).get("segment_stride_seconds", 15)
            segments = segment_audio(audio_data, sr, call_id, seg_duration, seg_stride)

            seg_list = []
            for seg in segments:
                # Mock or transcribe segment
                seg_list.append({
                    "segment_id": seg.segment_id,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "duration": seg.duration,
                    "annotation_type": seg.annotation_type,
                    "fraud_probability": pred_result.fraud_probability,
                })
            pred_result.segments = seg_list
        except Exception as seg_exc:
            logger.warning("Segment analysis encountered error: %s", seg_exc)

    # Format JSON
    json_output = result_to_json(pred_result)
    print("\n" + json.dumps(json_output, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    """Launch the FastAPI server."""
    import uvicorn

    host = getattr(args, "host", None) or config.get("api", {}).get("host", "0.0.0.0")
    port = getattr(args, "port", None) or config.get("api", {}).get("port", 8000)

    print("=" * 60)
    print(f"LAUNCHING FASTAPI SERVER on http://{host}:{port}")
    print("=" * 60)
    uvicorn.run("src.api.main:app", host=host, port=port, reload=False)
    return 0


# ---------------------------------------------------------------------------
# Argument Parser Setup
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct command line interface argument parser."""
    parser = argparse.ArgumentParser(
        prog="fraud-call-detection",
        description="AI-Based Fraudulent Phone Call Detection System",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_PROJECT_ROOT / "config.yaml"),
        help="Path to config.yaml (default: %(default)s)",
    )

    subparsers = parser.add_subparsers(dest="command", help="System command to execute")

    # download-data
    p_download = subparsers.add_parser("download-data", help="Download NCSU robocall dataset")
    p_download.add_argument("--max-files", type=int, default=None, help="Maximum audio files to download")
    p_download.add_argument("--force", action="store_true", help="Force redownload existing files")

    # validate-data
    subparsers.add_parser("validate-data", help="Validate audio and manifest files")

    # preprocess
    subparsers.add_parser("preprocess", help="Resample, mono-convert, and normalize audio")

    # transcribe
    p_transcribe = subparsers.add_parser("transcribe", help="Transcribe audio with Whisper")
    p_transcribe.add_argument("--model", type=str, default=None, help="Whisper model (tiny, base, small, medium)")
    p_transcribe.add_argument("--force", action="store_true", help="Re-transcribe even if cached")

    # prepare-dataset
    subparsers.add_parser("prepare-dataset", help="Build manifest and train/val/test splits")

    # train
    subparsers.add_parser("train", help="Train the BiLSTM classification model")

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate model on test split")
    p_eval.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Split to evaluate")

    # predict
    p_pred = subparsers.add_parser("predict", help="Predict fraud likelihood for an audio file")
    p_pred.add_argument("--audio", type=str, required=True, help="Path to audio file (WAV, MP3, etc.)")
    p_pred.add_argument("--segments", action="store_true", help="Include segment-level analysis")

    # serve
    p_serve = subparsers.add_parser("serve", help="Launch FastAPI server")
    p_serve.add_argument("--host", type=str, default=None, help="Host to bind (default: 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=None, help="Port to bind (default: 8000)")

    return parser


def main() -> int:
    """CLI main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        return 1

    config = load_config(config_path)
    config["project_root"] = str(_PROJECT_ROOT)
    _init_logging(config)

    commands = {
        "download-data": cmd_download_data,
        "validate-data": cmd_validate_data,
        "preprocess": cmd_preprocess,
        "transcribe": cmd_transcribe,
        "prepare-dataset": cmd_prepare_dataset,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "predict": cmd_predict,
        "serve": cmd_serve,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args, config)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
