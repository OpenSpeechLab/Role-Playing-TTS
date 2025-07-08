import os
import tempfile
from pathlib import Path
import logging
from flask import Flask, request, send_file, jsonify, after_this_request
from Audio import load_mmaudio_model, audio
from model import load_cosyvoice_model, tts
import werkzeug.utils
from rag import rag_speakers, last_token_pool, get_detailed_instruct
# RAG imports
import torch
import torch.nn.functional as F
import json5 as json
from torch import Tensor
from modelscope import AutoTokenizer, AutoModel

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)

# --- Model Loading ---
MODEL_VARIANT = 'large_44k_v2'
FULL_PRECISION = False
MODELS = {
    "mmaudio": None,
    "cosyvoice": None,
    "rag": None,
}



def load_rag_model(model_name: str = 'Qwen/Qwen3-Embedding-0.6B'):
    """Load RAG embedding model."""
    log.info(f"Loading RAG model from: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        model = AutoModel.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')
        log.info(f"RAG model loaded to device: {model.device}")
        return {"tokenizer": tokenizer, "model": model}
    except Exception as e:
        log.error(f"Error loading RAG model: {e}", exc_info=True)
        raise

def load_models():
    """Load all models in a consolidated way."""
    
    # Configuration for all models to be loaded
    model_loaders = {
        "mmaudio": lambda: load_mmaudio_model(variant=MODEL_VARIANT, full_precision=FULL_PRECISION),
        "cosyvoice": load_cosyvoice_model,
        "rag": load_rag_model
    }

    log.info("Loading all models...")
    for name, loader in model_loaders.items():
        if MODELS[name] is None:
            log.info(f"Loading {name} model...")
            try:
                MODELS[name] = loader()
                log.info(f"{name.capitalize()} model loaded successfully.")
            except Exception as e:
                log.error(f"Error loading {name} model: {e}", exc_info=True)
                # Continue to allow other models to load

# --- Routes ---
@app.route('/audio', methods=['POST'])
def generate_audio():
    if MODELS["mmaudio"] is None:
        return jsonify({"error": "MMAudio model is not loaded. Please try again later."}), 503

    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({"error": "'prompt' is a required field."}), 400

        # Extract parameters from request
        prompt = data.get('prompt')
        negative_prompt = data.get('negative_prompt', '')
        duration = float(data.get('duration', 8.0))
        cfg_strength = float(data.get('cfg_strength', 4.5))
        num_steps = int(data.get('num_steps', 100))
        seed = int(data.get('seed', 42))
        normalize = bool(data.get('normalize', True))
        volume = float(data.get('volume', -23.0))
        peak_norm_db = float(data.get('peak_norm_db_for_norm', -1.0))

        # Generate audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            output_path = Path(tmp_file.name)

        log.info(f"Generating audio for prompt: {prompt}")

        generated_path = audio(
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_bundle=MODELS["mmaudio"],
            duration=duration,
            cfg_strength=cfg_strength,
            num_steps=num_steps,
            seed=seed,
            output_path=output_path,
            normalize=normalize,
            volume=volume,
            peak_norm_db_for_norm=peak_norm_db,
        )

        log.info(f"Audio generated at: {generated_path}")
        
        # Clean up temporary files after the request
        @after_this_request
        def cleanup(response):
            try:
                os.remove(generated_path)
                log.info(f"Cleaned up temporary file: {generated_path}")
            except OSError as e:
                log.error(f"Error cleaning up file {generated_path}: {e}", exc_info=True)
            return response

        return send_file(
            generated_path,
            as_attachment=True,
            download_name=f'generated_audio.wav',
            mimetype='audio/wav'
        )

    except Exception as e:
        log.error(f"An error occurred during audio generation: {e}", exc_info=True)
        return jsonify({"error": "Failed to generate audio.", "details": str(e)}), 500


@app.route('/tts', methods=['POST'])
def generate_tts():
    if MODELS["cosyvoice"] is None:
        return jsonify({"error": "CosyVoice model is not loaded. Please try again later."}), 503

    try:
        # Check for required form fields
        if 'tts_text' not in request.form or 'prompt_text' not in request.form:
            return jsonify({"error": "'tts_text' and 'prompt_text' are required form fields."}), 400
        
        # Check for required file
        if 'prompt_speech_file' not in request.files:
            return jsonify({"error": "'prompt_speech_file' is a required file upload."}), 400

        tts_text = request.form['tts_text']
        prompt_text = request.form['prompt_text']
        prompt_speech_file = request.files['prompt_speech_file']
        
        # Optional parameters
        speed = float(request.form.get('speed', 1.0))
        normalize = request.form.get('normalize', 'true').lower() == 'true'
        volume = float(request.form.get('volume', -23.0))
        peak_norm_db = float(request.form.get('peak_norm_db_for_norm', -1.0))

        # Save the uploaded prompt speech to a temporary file
        prompt_filename = werkzeug.utils.secure_filename(prompt_speech_file.filename)
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(prompt_filename)[1], delete=False) as tmp_prompt_file:
            prompt_speech_file.save(tmp_prompt_file.name)
            prompt_speech_path = tmp_prompt_file.name

        # Prepare output path in a temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_output_file:
            output_path = Path(tmp_output_file.name)

        log.info(f"Generating TTS for text: {tts_text[:50]}...")

        tts(
            model=MODELS["cosyvoice"],
            tts_text=tts_text,
            prompt_text=prompt_text,
            prompt_speech_16k=prompt_speech_path,
            out_wav=str(output_path),
            speed=speed,
            normalize=normalize,
            volume=volume,
            peak_norm_db_for_norm=peak_norm_db
        )

        log.info(f"TTS audio generated at: {output_path}")

        # Clean up temporary files after the request
        @after_this_request
        def cleanup_tts(response):
            try:
                os.remove(prompt_speech_path)
                log.info(f"Cleaned up temporary prompt file: {prompt_speech_path}")
                os.remove(output_path)
                log.info(f"Cleaned up temporary output file: {output_path}")
            except OSError as e:
                log.error(f"Error cleaning up TTS files: {e}", exc_info=True)
            return response

        return send_file(
            output_path,
            as_attachment=True,
            download_name='generated_tts.wav',
            mimetype='audio/wav'
        )

    except Exception as e:
        log.error(f"An error occurred during TTS generation: {e}", exc_info=True)
        # Clean up temporary files if they exist on error
        if 'prompt_speech_path' in locals() and os.path.exists(prompt_speech_path):
            os.remove(prompt_speech_path)
        if 'output_path' in locals() and os.path.exists(output_path):
            os.remove(output_path)
        return jsonify({"error": "Failed to generate TTS audio.", "details": str(e)}), 500

@app.route('/rag_speakers', methods=['POST'])
def run_rag_speakers():
    if MODELS["rag"] is None:
        return jsonify({"error": "RAG model is not loaded. Please try again later."}), 503

    try:
        if 'query_file' not in request.files or 'doc_file' not in request.files:
            return jsonify({"error": "'query_file' and 'doc_file' are required file uploads."}), 400

        query_file = request.files['query_file']
        doc_file = request.files['doc_file']

        # Save uploaded files to temporary files
        with tempfile.NamedTemporaryFile(delete=False, mode='wb') as tmp_query_file:
            query_file.save(tmp_query_file.name)
            query_path = tmp_query_file.name
        
        with tempfile.NamedTemporaryFile(delete=False, mode='wb') as tmp_doc_file:
            doc_file.save(tmp_doc_file.name)
            doc_path = tmp_doc_file.name
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_output_file:
            output_json_path = tmp_output_file.name

        log.info(f"Running RAG for query file: {query_file.filename} and doc file: {doc_file.filename}")

        result = rag_speakers(
            query_jsonl_path=query_path,
            doc_json_path=doc_path,
            output_json_path=output_json_path,
            tokenizer=MODELS["rag"]['tokenizer'],
            model=MODELS["rag"]['model']
        )
        
        # Cleanup and send file
        @after_this_request
        def cleanup_rag(response):
            try:
                os.remove(query_path)
                os.remove(doc_path)
                os.remove(output_json_path)
                log.info(f"Cleaned up temporary RAG files: {query_path}, {doc_path}, {output_json_path}")
            except OSError as e:
                log.error(f"Error cleaning up RAG files: {e}", exc_info=True)
            return response
            
        return send_file(
            output_json_path,
            as_attachment=True,
            download_name='rag_match_results.json',
            mimetype='application/json'
        )

    except Exception as e:
        log.error(f"An error occurred during RAG execution: {e}", exc_info=True)
        # Clean up temporary files on error
        if 'query_path' in locals() and os.path.exists(query_path):
            os.remove(query_path)
        if 'doc_path' in locals() and os.path.exists(doc_path):
            os.remove(doc_path)
        if 'output_json_path' in locals() and os.path.exists(output_json_path):
            os.remove(output_json_path)
        return jsonify({"error": "Failed to run RAG.", "details": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify that the service is running."""
    audio_model_loaded = MODELS["mmaudio"] and MODELS["mmaudio"].get('net')
    cosyvoice_model_loaded = MODELS["cosyvoice"] is not None
    rag_model_loaded = MODELS["rag"] is not None
    
    models_status = {
        "mmaudio": bool(audio_model_loaded), 
        "cosyvoice": cosyvoice_model_loaded,
        "rag": rag_model_loaded
    }
    
    all_models_ok = all(models_status.values())

    if all_models_ok:
        return jsonify({"status": "ok", "models_loaded": models_status}), 200
    else:
        return jsonify({"status": "partially_loaded" if any(models_status.values()) else "error", "models_loaded": models_status}), 503

if __name__ == '__main__':
    load_models()  # Load all models before starting the server
    app.run(host='0.0.0.0', port=8000, debug=False)
