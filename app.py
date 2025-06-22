# DebateLens - Analisi Comparativa della Comunicazione
# Craicek's Version - Rizzo AI Academy
# Versione Finale Produzione

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
import base64
import sys
from datetime import datetime
import traceback
import logging

# Caricamento variabili ambiente
from dotenv import load_dotenv

load_dotenv()

# Import AI e visualizzazione
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from io import BytesIO


@dataclass
class AnalysisResult:
    """Risultato dell'analisi di un singolo partecipante"""
    participant_name: str
    rigorosita_tecnica: int
    uso_dati_oggettivi: int
    approccio_divulgativo: int
    stile_comunicativo: int
    focalizzazione_argomento: int
    orientamento_pratico: int
    explanations: dict

    def to_dict(self) -> dict:
        return {
            'participant_name': self.participant_name,
            'scores': {
                'rigorosita_tecnica': self.rigorosita_tecnica,
                'uso_dati_oggettivi': self.uso_dati_oggettivi,
                'approccio_divulgativo': self.approccio_divulgativo,
                'stile_comunicativo': self.stile_comunicativo,
                'focalizzazione_argomento': self.focalizzazione_argomento,
                'orientamento_pratico': self.orientamento_pratico
            },
            'explanations': self.explanations
        }


class DebateLensAnalyzer:
    """Core analyzer per DebateLens con Google Gemini AI"""

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.last_error = None
        self.analysis_criteria = [
            'rigorosita_tecnica', 'uso_dati_oggettivi', 'approccio_divulgativo',
            'stile_comunicativo', 'focalizzazione_argomento', 'orientamento_pratico'
        ]

    def create_analysis_prompt(self, text: str, participant_name: str) -> str:
        """Crea il prompt per l'analisi AI"""
        return f"""
Analizza il seguente testo di {participant_name} secondo questi 6 criteri, assegnando un punteggio da 1 a 10:

TESTO: {text}

CRITERI:
1. Rigorosità tecnica (1-10): Precisione terminologica e concetti specialistici
2. Uso di dati oggettivi (1-10): Statistiche, ricerche, fonti verificabili
3. Approccio divulgativo (1-10): Accessibilità, esempi, analogie
4. Stile comunicativo (1-10): Fluidità e capacità di coinvolgimento
5. Focalizzazione argomento (1-10): Aderenza al tema, coerenza logica
6. Orientamento pratico (1-10): Soluzioni concrete, applicabilità

FORMATO JSON:
{{
  "rigorosita_tecnica": X,
  "uso_dati_oggettivi": X,
  "approccio_divulgativo": X,
  "stile_comunicativo": X,
  "focalizzazione_argomento": X,
  "orientamento_pratico": X,
  "explanations": {{
    "rigorosita_tecnica": "Breve spiegazione",
    "uso_dati_oggettivi": "Breve spiegazione",
    "approccio_divulgativo": "Breve spiegazione",
    "stile_comunicativo": "Breve spiegazione",
    "focalizzazione_argomento": "Breve spiegazione",
    "orientamento_pratico": "Breve spiegazione"
  }}
}}
"""

    def analyze_participant(self, text: str, participant_name: str) -> AnalysisResult:
        """Analizza un partecipante con Google Gemini"""
        try:
            prompt = self.create_analysis_prompt(text, participant_name)

            generation_config = genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1000,
                response_mime_type="application/json"
            )

            response = self.model.generate_content(prompt, generation_config=generation_config)
            result_json = json.loads(response.text)

            return AnalysisResult(
                participant_name=participant_name,
                rigorosita_tecnica=result_json['rigorosita_tecnica'],
                uso_dati_oggettivi=result_json['uso_dati_oggettivi'],
                approccio_divulgativo=result_json['approccio_divulgativo'],
                stile_comunicativo=result_json['stile_comunicativo'],
                focalizzazione_argomento=result_json['focalizzazione_argomento'],
                orientamento_pratico=result_json['orientamento_pratico'],
                explanations=result_json['explanations']
            )

        except Exception as e:
            self.last_error = str(e)
            return None

    def create_radar_chart(self, analyses: list) -> str:
        """Genera radar chart PNG Iron Man style"""
        if not analyses:
            return None

        try:
            # Setup matplotlib per server
            import matplotlib
            matplotlib.use('Agg')
            plt.ioff()
            plt.clf()
            plt.close('all')

            # Crea il plot
            fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'), dpi=150)
            fig.patch.set_facecolor('#1a1a1a')

            # Labels e angoli
            labels = ['Rigorosità\nTecnica', 'Uso Dati\nOggettivi', 'Approccio\nDivulgativo',
                      'Stile\nComunicativo', 'Focalizzazione\nArgomento', 'Orientamento\nPratico']
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            angles += angles[:1]

            # Colori Iron Man
            colors = ['#dc2626', '#fbbf24', '#b91c1c', '#f59e0b']

            # Plot ogni partecipante
            for i, analysis in enumerate(analyses):
                values = [
                    analysis.rigorosita_tecnica, analysis.uso_dati_oggettivi,
                    analysis.approccio_divulgativo, analysis.stile_comunicativo,
                    analysis.focalizzazione_argomento, analysis.orientamento_pratico
                ]
                values += values[:1]

                color = colors[i % len(colors)]
                ax.plot(angles, values, 'o-', linewidth=4, label=analysis.participant_name,
                        color=color, markersize=10, markerfacecolor=color,
                        markeredgecolor='white', markeredgewidth=2)
                ax.fill(angles, values, alpha=0.25, color=color)

            # Styling Iron Man
            ax.set_facecolor('#0a0a0a')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontsize=12, color='#fbbf24', fontweight='bold')
            ax.set_ylim(0, 10)
            ax.set_yticks(range(0, 11, 2))
            ax.set_yticklabels(range(0, 11, 2), fontsize=10, color='#dc2626', fontweight='bold')
            ax.grid(True, color='#444444', alpha=0.7, linewidth=1)

            plt.title('🔥 DebateLens - Analisi Comparativa\nCraicek\'s Version',
                      size=18, fontweight='bold', pad=30, color='#fbbf24')

            legend = plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0),
                                facecolor='#1a1a1a', edgecolor='#dc2626',
                                labelcolor='#fbbf24', fontsize=11, framealpha=0.9)
            legend.get_frame().set_linewidth(2)

            plt.tight_layout()

            # Converti in base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                        facecolor='#1a1a1a', edgecolor='none', pad_inches=0.2)
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode()

            plt.close(fig)
            plt.clf()

            return chart_base64

        except Exception as e:
            logging.error(f"Error generating matplotlib radar chart: {e}", exc_info=True)
            return self.create_html_fallback(analyses)

    def create_html_fallback(self, analyses):
        """Fallback HTML quando matplotlib fallisce"""
        try:
            colors = ['#dc2626', '#fbbf24', '#b91c1c', '#f59e0b']

            html = '<div style="background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); padding: 40px; border-radius: 16px; text-align: center; border: 2px solid #dc2626;">'
            html += '<h3 style="color: #fbbf24; margin-bottom: 10px;">📊 Fallback Radar Visualization</h3>'
            html += '<p style="color: #ccc; font-size: 0.9em; margin-bottom: 20px;">Matplotlib chart generation failed. This is a stylized placeholder. Please refer to the detailed scores in the report.</p>'

            # Radar visuale
            html += '<div style="position: relative; width: 400px; height: 400px; margin: 0 auto; border: 3px solid #dc2626; border-radius: 50%; background: radial-gradient(circle, rgba(220, 38, 38, 0.1) 0%, rgba(0, 0, 0, 0.4) 100%);">'

            # Cerchi concentrici
            for i, radius in enumerate([20, 35, 50, 65, 80]):
                opacity = 0.6 - (i * 0.1)
                html += f'<div style="position: absolute; top: {50 - radius / 2}%; left: {50 - radius / 2}%; width: {radius}%; height: {radius}%; border: 1px solid #666; border-radius: 50%; opacity: {opacity};"></div>'

            # Assi
            html += '<div style="position: absolute; top: 0; left: 50%; width: 2px; height: 100%; background: linear-gradient(to bottom, #dc2626, #fbbf24, #dc2626); opacity: 0.8;"></div>'
            html += '<div style="position: absolute; top: 50%; left: 0; width: 100%; height: 2px; background: linear-gradient(to right, #dc2626, #fbbf24, #dc2626); opacity: 0.8;"></div>'

            # Punti dati
            for i, analysis in enumerate(analyses[:4]):
                color = colors[i % len(colors)]
                html += f'<div style="position: absolute; top: {20 + i * 5}%; left: {60 + i * 5}%; width: 12px; height: 12px; background: {color}; border-radius: 50%; border: 2px solid white; z-index: 10;" title="{analysis.participant_name}"></div>'

            html += '<div style="position: absolute; top: 50%; left: 50%; width: 8px; height: 8px; background: #fbbf24; border-radius: 50%; transform: translate(-50%, -50%); border: 2px solid white;"></div>'
            html += '</div>'

            # Legenda
            html += '<div style="margin-top: 25px; display: flex; justify-content: center; gap: 20px;">'
            for i, analysis in enumerate(analyses[:4]):
                color = colors[i % len(colors)]
                html += f'<div style="display: flex; align-items: center; gap: 8px;"><div style="width: 16px; height: 16px; background: {color}; border-radius: 50%; border: 2px solid white;"></div><span style="color: #fbbf24; font-weight: bold;">{analysis.participant_name}</span></div>'
            html += '</div></div>'

            return html

        except Exception as e:
            return None

    def generate_comparative_report(self, analyses: list) -> dict:
        """Genera report comparativo con insights"""
        if len(analyses) < 2:
            return {"error": "Servono almeno 2 partecipanti per il confronto"}

        report = {
            "summary": {},
            "detailed_comparison": {},
            "insights": []
        }

        try:
            # Calcola statistiche per criterio
            for criterion in self.analysis_criteria:
                scores = [getattr(analysis, criterion) for analysis in analyses]
                report["summary"][criterion] = {
                    "average": round(sum(scores) / len(scores), 2),
                    "max_participant": analyses[scores.index(max(scores))].participant_name,
                    "max_score": max(scores),
                    "min_participant": analyses[scores.index(min(scores))].participant_name,
                    "min_score": min(scores)
                }

            # Dettagli partecipanti
            for analysis in analyses:
                report["detailed_comparison"][analysis.participant_name] = analysis.to_dict()

            # Insights automatici potenziati
            report["insights"].append(f"📊 Analisi completata per {len(analyses)} partecipanti.")

            if len(analyses) >= 1: # Need at least one for some insights
                for criterion in self.analysis_criteria:
                    criterion_label = criterion.replace("_", " ").capitalize()

                    # Get all scores for the current criterion, ensuring they are integers
                    scores = []
                    for analysis in analyses:
                        score_value = getattr(analysis, criterion, None)
                        if score_value is not None:
                            try:
                                scores.append(int(score_value))
                            except ValueError:
                                # Handle case where score might not be convertible to int, though dataclass types should ensure this
                                logging.warning(f"Could not convert score for {criterion} for participant {analysis.participant_name} to int.")
                                scores.append(0) # Default or skip
                        else:
                            scores.append(0) # Default if attribute somehow missing

                    if not scores: # Should not happen if analyses is not empty
                        continue

                    max_score = -1
                    min_score = 11 # Scores are 1-10

                    # Recalculate max_score and min_score based on actual scores present
                    # This avoids issues if all scores are 0 due to errors or missing data
                    if any(s > 0 for s in scores): # Check if there are any actual scores
                        max_score = max(s for s in scores if s is not None)
                        min_score = min(s for s in scores if s is not None and s > 0) # Min of actual scores
                    else: # All scores are 0 or None
                        max_score = 0
                        min_score = 0


                    best_performers = [analyses[i].participant_name for i, score in enumerate(scores) if score == max_score and max_score > 0]
                    worst_performers = [analyses[i].participant_name for i, score in enumerate(scores) if score == min_score and max_score > min_score] # Only show if there's a difference from max

                    if best_performers:
                        if len(best_performers) == len(analyses) and len(analyses) > 1 and max_score > 0 : # All are equally best
                             report["insights"].append(f"🏆 Tutti i partecipanti mostrano un punteggio massimo ({max_score}/10) in {criterion_label}.")
                        elif len(best_performers) == 1:
                            report["insights"].append(f"🥇 {best_performers[0]} eccelle in {criterion_label} ({max_score}/10).")
                        elif len(best_performers) > 1 :
                             report["insights"].append(f"🥇 {', '.join(best_performers)} guidano in {criterion_label} ({max_score}/10).")

                    if worst_performers and len(worst_performers) < len(analyses): # Don't show if all are 'worst' (e.g. all got min score)
                        if len(worst_performers) == 1:
                             report["insights"].append(f"🔻 {worst_performers[0]} ha il punteggio più basso in {criterion_label} ({min_score}/10).")
                        elif len(worst_performers) > 1:
                             report["insights"].append(f"🔻 {', '.join(worst_performers)} hanno i punteggi più bassi in {criterion_label} ({min_score}/10).")

                # Example of pairwise comparison (can be extensive if many participants)
                if len(analyses) == 2:
                    p1, p2 = analyses[0], analyses[1]
                    p1_tech = int(p1.rigorosita_tecnica)
                    p2_tech = int(p2.rigorosita_tecnica)
                    if abs(p1_tech - p2_tech) <= 1:
                        report["insights"].append("🎯 Livello tecnico equilibrato tra i due partecipanti.")
                    elif p1_tech > p2_tech:
                        report["insights"].append(f"🔥 {p1.participant_name} è più forte in rigorosità tecnica rispetto a {p2.participant_name}.")
                    else:
                        report["insights"].append(f"🔥 {p2.participant_name} è più forte in rigorosità tecnica rispetto a {p1.participant_name}.")


        except Exception as e:
            logging.error(f"Errore nella generazione del report comparativo: {e}", exc_info=True)
            # Fallback insights if there was an error during report generation itself
            report["insights"] = [
                "⚠️ Si è verificato un errore durante la generazione degli insights dettagliati.",
                f"📊 {len(analyses)} partecipanti sono stati comunque processati per i punteggi base."
            ]

        return report


def analyze_text_heuristic(text):
    """Analisi euristica per fallback senza AI"""
    if not text:
        return {criterion: 5 for criterion in ['rigorosita_tecnica', 'uso_dati_oggettivi',
                                               'approccio_divulgativo', 'stile_comunicativo',
                                               'focalizzazione_argomento', 'orientamento_pratico']}

    text_lower = text.lower()
    word_count = len(text.split())

    # Parole chiave per analisi
    technical_words = ['analisi', 'sistema', 'processo', 'metodologia', 'implementazione']
    data_words = ['percentuale', 'statistica', 'dati', 'ricerca', 'studio', '%']
    divulgative_words = ['semplice', 'facile', 'esempio', 'immaginate', 'praticamente']
    practical_words = ['utilizzare', 'applicare', 'soluzione', 'problema', 'pratico']

    # Calcola scores
    tech_score = min(10, 4 + sum(1 for word in technical_words if word in text_lower))
    data_score = min(10, 3 + sum(1 for word in data_words if word in text_lower) * 2)
    divulgative_score = min(10, 4 + sum(1 for word in divulgative_words if word in text_lower))
    style_score = min(10, 5 + min(3, word_count // 50))
    focus_score = max(3, min(10, 8 - text.count('?') - text.count('...')))
    practical_score = min(10, 4 + sum(1 for word in practical_words if word in text_lower))

    return {
        'rigorosita_tecnica': tech_score,
        'uso_dati_oggettivi': data_score,
        'approccio_divulgativo': divulgative_score,
        'stile_comunicativo': style_score,
        'focalizzazione_argomento': focus_score,
        'orientamento_pratico': practical_score
    }


def create_heuristic_analysis(name, text):
    """Crea AnalysisResult usando analisi euristica"""
    scores = analyze_text_heuristic(text)

    explanations = {
        'rigorosita_tecnica': f'{name} {"usa terminologia tecnica appropriata" if scores["rigorosita_tecnica"] >= 7 else "utilizza un linguaggio più generale"}',
        'uso_dati_oggettivi': f'{"Presenta riferimenti a dati" if scores["uso_dati_oggettivi"] >= 6 else "Basato più su opinioni"}',
        'approccio_divulgativo': f'{"Stile molto accessibile" if scores["approccio_divulgativo"] >= 7 else "Approccio più tecnico"}',
        'stile_comunicativo': f'Comunicazione {["essenziale", "buona", "efficace", "eccellente"][min(3, scores["stile_comunicativo"] // 3)]}',
        'focalizzazione_argomento': f'Mantiene {"buon" if scores["focalizzazione_argomento"] >= 6 else "discreto"} focus',
        'orientamento_pratico': f'Orientamento {"molto" if scores["orientamento_pratico"] >= 7 else "moderatamente"} pratico'
    }

    return AnalysisResult(
        participant_name=name,
        rigorosita_tecnica=scores['rigorosita_tecnica'],
        uso_dati_oggettivi=scores['uso_dati_oggettivi'],
        approccio_divulgativo=scores['approccio_divulgativo'],
        stile_comunicativo=scores['stile_comunicativo'],
        focalizzazione_argomento=scores['focalizzazione_argomento'],
        orientamento_pratico=scores['orientamento_pratico'],
        explanations=explanations
    )


# Configurazione
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY') # No string default from getenv

analyzer = None # Default to None
AI_MODE = False # Default to False

if GOOGLE_API_KEY and GOOGLE_API_KEY != 'your-google-api-key-here' and GOOGLE_API_KEY.strip() != "":
    try:
        analyzer = DebateLensAnalyzer(GOOGLE_API_KEY)
        AI_MODE = True
        logging.info("Google Gemini AI configured successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize DebateLensAnalyzer with API key: {e}", exc_info=True)
        # analyzer remains None, AI_MODE remains False
        # The print statements at startup will reflect that AI is not configured.
else:
    # This else block covers cases where API_KEY is None, the placeholder string, or an empty string.
    # analyzer remains None, AI_MODE remains False
    # No need to instantiate DebateLensAnalyzer with "fake-key" if it won't be used.
    logging.info("Google API Key not provided or is placeholder. Using heuristic mode.")


# Flask App
app = Flask(__name__)
CORS(app)

# Imports for YouTube Transcription
import yt_dlp # For downloading YouTube audio
import whisper # For audio transcription
import tempfile # For handling temporary audio files
# os is already imported, but ensure it's available for file deletion if needed by tempfile explicitly.

# Global variable for Whisper model (load once)
# Loading the model can be time-consuming, so we should do it once when the app starts.
# For simplicity in this step, I might load it within the request first,
# but a production app should load it globally or on first request.
# Let's try loading it globally but handle potential errors.
WHISPER_MODEL = None
try:
    WHISPER_MODEL = whisper.load_model("base") # Using the base model for now
    logging.info("Whisper model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading Whisper model: {e}", exc_info=True)
    WHISPER_MODEL = None # Ensure it's None if loading failed


# Routes
@app.route('/')
def serve_frontend():
    return send_from_directory('static', 'index.html')


# Removed the insecure route:
# @app.route('/<path:filename>')
# def serve_static(filename):
#     return send_from_directory('.', filename)


@app.route('/api/analyze', methods=['POST'])
def analyze_debate():
    """Endpoint principale per l'analisi dei dibattiti"""
    try:
        data = request.get_json()

        if not data or len(data.get('participants', [])) < 2:
            return jsonify({'error': 'Servono almeno 2 partecipanti'}), 400

        participants_data = data['participants']
        analyses = []
        participant_analysis_details = [] # Stores details about how each participant was analyzed

        MAX_TEXT_LENGTH = 10000 # Max characters per participant text

        # Valida e analizza ogni partecipante
        for p_data in participants_data:
            name = p_data.get('name', 'Partecipante')
            text = p_data.get('text', '')
            analysis_status = {"name": name, "type": "heuristic", "error": None}

            if not text.strip():
                analysis_status["error"] = "Testo vuoto fornito."
                # analyses.append(create_heuristic_analysis(name, "")) # Or skip adding to main analyses list
                participant_analysis_details.append(analysis_status)
                continue

            if len(text) > MAX_TEXT_LENGTH:
                # This case is already handled and returns, so won't be hit if previous check is active.
                # However, if we were to collect all errors first, this would be relevant.
                # For now, the immediate return is fine.
                return jsonify({'error': f"Il testo per '{name}' supera la lunghezza massima di {MAX_TEXT_LENGTH} caratteri."}), 400

            if AI_MODE:
                try:
                    analysis_obj = analyzer.analyze_participant(text, name)
                    if analysis_obj:
                        analyses.append(analysis_obj)
                        analysis_status["type"] = "ai"
                    else: # AI analysis failed, analyzer.last_error should be set
                        analyses.append(create_heuristic_analysis(name, text))
                        analysis_status["error"] = analyzer.last_error or "Analisi AI fallita, fallback a euristica."
                        analyzer.last_error = None # Reset last_error after consuming it
                except Exception as e_inner:
                    analyses.append(create_heuristic_analysis(name, text))
                    analysis_status["error"] = f"Eccezione durante analisi AI: {str(e_inner)}"
            else:
                analyses.append(create_heuristic_analysis(name, text))

            participant_analysis_details.append(analysis_status)

        if len(analyses) < 2:
            # This check might need adjustment if empty texts are skipped from `analyses` list
            # but are present in `participant_analysis_details`.
            # For now, assuming `analyses` must have at least 2 valid entries.
            return jsonify({
                'error': 'Analisi fallita o partecipanti insufficienti con testo valido.',
                'participant_analysis_status': participant_analysis_details
            }), 500

        # Genera risultati
        chart_data = analyzer.create_radar_chart(analyses)
        report = analyzer.generate_comparative_report(analyses)

        return jsonify({
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'participants_count': len(analyses),
            'chart_data': chart_data,
            'report': report,
            'version': 'DebateLens Craicek\'s Version',
            'ai_mode': AI_MODE,
            'participant_analysis_status': participant_analysis_details
        })

    except Exception as e:
        return jsonify({'error': f'Errore: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'version': 'DebateLens Craicek\'s Version',
        'ai_mode': AI_MODE,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/transcribe_youtube', methods=['POST'])
def transcribe_youtube_audio():
    if WHISPER_MODEL is None:
        return jsonify({'error': 'Whisper model not loaded. Transcription unavailable.'}), 503

    data = request.get_json()
    youtube_url = data.get('youtube_url')

    if not youtube_url:
        return jsonify({'error': 'YouTube URL not provided.'}), 400

    temp_audio_file = None
    try:
        # 1. Download Audio using yt-dlp
        # Create a temporary file to store the audio
        # We need a named temporary file that yt-dlp can write to and whisper can read.
        # tempfile.NamedTemporaryFile can be tricky with reopening on some OS,
        # so constructing path with tempfile.gettempdir() and a unique name is safer.

        temp_dir = tempfile.gettempdir()
        # Create a unique filename. Using a simple approach for now.
        # For production, consider using uuid for truly unique names.
        temp_filename = f"youtube_audio_{os.urandom(8).hex()}.mp3"
        temp_audio_path = os.path.join(temp_dir, temp_filename)

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': temp_audio_path, # Output template for the filename
            'noplaylist': True,
            'quiet': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3', # Output format
                'preferredquality': '192', # Bitrate
            }],
        }

        logging.info(f"Attempting to download audio from: {youtube_url} to {temp_audio_path}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        # Check if file was downloaded (yt-dlp might not raise error for some failed downloads)
        if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
            logging.error(f"Audio download failed or resulted in an empty file for URL: {youtube_url}")
            return jsonify({'error': 'Failed to download audio from the YouTube URL. The video might be private, unavailable, or the URL is incorrect.'}), 500

        logging.info(f"Audio downloaded successfully: {temp_audio_path}")
        temp_audio_file = temp_audio_path # Keep track for deletion

        # 2. Transcribe Audio using Whisper
        logging.info(f"Starting transcription for {temp_audio_file}...")
        result = WHISPER_MODEL.transcribe(temp_audio_file, fp16=False) # fp16=False for CPU
        transcribed_text = result['text']
        logging.info("Transcription complete.")

        return jsonify({'transcript': transcribed_text})

    except yt_dlp.utils.DownloadError as e:
        logging.error(f"yt-dlp DownloadError for URL {youtube_url}: {e}", exc_info=True)
        return jsonify({'error': f'Error downloading video: {str(e).splitlines()[-1] if str(e) else "Unknown yt-dlp error"}'}), 500
    except Exception as e:
        logging.error(f"Error during YouTube transcription for URL {youtube_url}: {e}", exc_info=True)
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
    finally:
        # 3. Cleanup: Delete the temporary audio file
        if temp_audio_file and os.path.exists(temp_audio_file):
            try:
                os.remove(temp_audio_file)
                logging.info(f"Temporary audio file {temp_audio_file} deleted.")
            except Exception as e_del:
                logging.error(f"Error deleting temporary audio file {temp_audio_file}: {e_del}", exc_info=True)


if __name__ == '__main__':
    print("🔥" * 20)
    print("🎯 DebateLens - Craicek's Version")
    print("🚀 Rizzo AI Academy")
    print("🔥" * 20)

    if AI_MODE:
        print("✅ Google Gemini AI configurato")
        print("🤖 Modalità: Analisi AI completa")
    else:
        print("🧠 Modalità: Analisi euristica intelligente")
        print("💡 Configura GOOGLE_API_KEY nel file .env per AI completa")

    print("🌐 Server: http://localhost:5000")
    print("=" * 60)

    app.run(debug=False, host='0.0.0.0', port=5000)