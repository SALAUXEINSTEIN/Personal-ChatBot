"""
Gradio deployment interface — Section 3.7.

Implements the four required interface elements:
    1. Chat window with alternating user/system message styling
    2. User profile panel showing the system's inferred preferences
       (editable by the participant)
    3. Persona-mode toggle for within-session A/B comparison
    4. Per-response feedback button feeding the human evaluation study

Also includes the persistent ethical disclaimer required by Section 3.9.5.

Usage (after training, or with a base pretrained model for a quick demo):
    python -m app.gradio_app --backbone_checkpoint checkpoints/stage1_backbone \
                              --stage2_checkpoint checkpoints/stage2_full_system
"""

from __future__ import annotations
import argparse
import csv
import os
import socket
import sys
from dataclasses import dataclass
from datetime import datetime

import gradio as gr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class SimpleDialogueState:
    current_topic: str = ""
    current_intent: str = ""
    unresolved_needs: list = None
    formality_score: float = 0.5
    topic_interests: dict = None
    emotional_tone: str = ""
    explicit_preferences: list = None

    def __post_init__(self):
        if self.unresolved_needs is None:
            self.unresolved_needs = []
        if self.topic_interests is None:
            self.topic_interests = {}
        if self.explicit_preferences is None:
            self.explicit_preferences = []

    def update(self, dst_prediction: dict, alpha: float = None):
        self.current_topic = dst_prediction.get("topic", self.current_topic)
        self.current_intent = dst_prediction.get("dialogue_act", self.current_intent)
        self.emotional_tone = dst_prediction.get("sentiment", self.emotional_tone)
        self.formality_score = 0.5 if dst_prediction.get("formality") == "formal" else 0.5


DialogueState = SimpleDialogueState

class StubPersonalisedChatbotSystem:
    """Lightweight stub system for quick demo runs without heavy model downloads."""
    def __init__(self):
        self.use_upe = False
        self.upe = None
        self.dst = None
        self.backbone = None

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate_response(self, persona_sentences, raw_history, dialogue_state=None):
        # Return a deterministic canned reply using last user message
        last = raw_history[-1] if raw_history else "Hello"
        return f"Demo reply to: {last}"


DISCLAIMER = (
    "⚠️ **Research Prototype Notice**: You are interacting with an academic AI research "
    "system (MSc dissertation project). Responses may contain errors, inconsistencies, or "
    "biases, and this system is **not** designed for sensitive applications (e.g. medical, "
    "legal, or crisis support). No automated decisions are made based on its outputs. "
    "Participation in feedback collection is voluntary and anonymised (Section 3.9)."
)

FEEDBACK_LOG_PATH = "./app/feedback_log.csv"


def build_system(backbone_checkpoint: str, stage2_checkpoint: str = None, device: str = None):
    import torch
    from models.backbone import load_backbone_and_tokenizer, load_finetuned_backbone
    from models.personalised_chatbot import PersonalisedChatbotSystem

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if backbone_checkpoint:
        backbone, tokenizer = load_finetuned_backbone(backbone_checkpoint, device)
    else:
        backbone, tokenizer = load_backbone_and_tokenizer()

    system = PersonalisedChatbotSystem(backbone, tokenizer, use_upe=True, use_dst=True).to(device)

    if stage2_checkpoint:
        system.upe.load_state_dict(
            torch.load(os.path.join(stage2_checkpoint, "upe.pt"), map_location=device))
        system.dst.load_state_dict(
            torch.load(os.path.join(stage2_checkpoint, "dst.pt"), map_location=device))

    system.eval()
    return system


def log_feedback(participant_code, session_id, turn_idx, rating, comment):
    os.makedirs(os.path.dirname(FEEDBACK_LOG_PATH), exist_ok=True)
    file_exists = os.path.isfile(FEEDBACK_LOG_PATH)
    with open(FEEDBACK_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "participant_code", "session_id", "turn_idx", "rating", "comment"])
        writer.writerow([datetime.utcnow().isoformat(), participant_code, session_id, turn_idx, rating, comment])
    return "Feedback recorded. Thank you."


def find_free_port(start_port: int = 7860, max_attempts: int = 10) -> int:
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise OSError(f"No free port found between {start_port} and {start_port + max_attempts - 1}")


def launch_app(system):
    with gr.Blocks(title="Personalised Transformer Chatbot — Research Prototype") as demo:
        gr.Markdown("## Personalised Transformer Chatbot (MSc Dissertation Research Prototype)")
        gr.Markdown(DISCLAIMER)

        state_history = gr.State([])                     # raw turn strings for the backbone
        state_dialogue_state = gr.State(DialogueState())
        state_persona = gr.State([])                      # list of persona sentences

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### 1. Chat window")
                chatbot_ui = gr.Chatbot(label="Conversation", height=420)
                msg_box = gr.Textbox(label="Your message", placeholder="Type a message and press Enter...")
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear conversation")

                gr.Markdown("### 4. Feedback")
                with gr.Row():
                    rating = gr.Radio(["1", "2", "3", "4", "5"], label="Rate the last response (1=poor, 5=excellent)")
                    feedback_comment = gr.Textbox(label="Optional comment", placeholder="Anything you'd like to note?")
                with gr.Row():
                    participant_code = gr.Textbox(label="Participant code (e.g. P001)", value="P000")
                    session_id = gr.Textbox(label="Session ID", value="session_1")
                feedback_btn = gr.Button("Submit feedback for last response")
                feedback_status = gr.Markdown("")

            with gr.Column(scale=1):
                gr.Markdown("### 2. User profile panel")
                persona_input = gr.Textbox(
                    label="Your persona sentences (one per line)",
                    placeholder="I enjoy hiking.\nI work as a teacher.\nI have two dogs.",
                    lines=5,
                )
                set_persona_btn = gr.Button("Set / update persona")

                gr.Markdown("**Inferred preferences (editable)**")
                inferred_topic = gr.Textbox(label="Current topic", interactive=True)
                inferred_formality = gr.Slider(0, 1, value=0.5, label="Formality (0=informal, 1=formal)")
                inferred_tone = gr.Textbox(label="Emotional tone", interactive=True)

                gr.Markdown("### 3. Persona mode")
                persona_toggle = gr.Checkbox(label="Enable persona-conditioned personalisation", value=True)

        def history_to_messages(history):
            return [
                {"role": "user" if idx % 2 == 0 else "assistant", "content": text}
                for idx, text in enumerate(history)
            ]

        def respond(user_message, history, dialogue_state, persona_sentences, persona_enabled):
            if not user_message:
                return (
                    history_to_messages(history),
                    history,
                    dialogue_state,
                    "",
                    "",
                    0.5,
                    "",
                )

            raw_history = history + [user_message] if isinstance(history, list) else [user_message]
            active_personas = persona_sentences if persona_enabled else []

            if dialogue_state is None:
                dialogue_state = DialogueState()

            if isinstance(system, StubPersonalisedChatbotSystem):
                reply = system.generate_response(active_personas, raw_history, dialogue_state=dialogue_state)
            else:
                system.use_upe = bool(persona_enabled)
                reply = system.generate_response(active_personas, raw_history, dialogue_state=dialogue_state)

                if system.use_dst and system.dst is not None:
                    try:
                        prediction = system.dst.predict_from_turns(
                            raw_history + [reply],
                            device=next(system.backbone.parameters()).device,
                        )
                        dialogue_state.update(prediction)
                    except Exception:
                        pass

            new_history = raw_history + [reply]
            display = history_to_messages(new_history)

            return (
                display,
                new_history,
                dialogue_state,
                dialogue_state.current_topic or "",
                dialogue_state.formality_score,
                dialogue_state.emotional_tone or "",
            )

        def set_persona(persona_text):
            sentences = [s.strip() for s in persona_text.split("\n") if s.strip()]
            return sentences

        def clear_conversation():
            return [], [], DialogueState(), "", 0.5, ""

        def submit_feedback(p_code, s_id, history, rating_val, comment):
            turn_idx = len(history) // 2
            return log_feedback(p_code, s_id, turn_idx, rating_val, comment)

        send_btn.click(
            respond,
            inputs=[msg_box, state_history, state_dialogue_state, state_persona, persona_toggle],
            outputs=[chatbot_ui, state_history, state_dialogue_state,
                     inferred_topic, inferred_formality, inferred_tone],
        ).then(lambda: "", None, msg_box)

        msg_box.submit(
            respond,
            inputs=[msg_box, state_history, state_dialogue_state, state_persona, persona_toggle],
            outputs=[chatbot_ui, state_history, state_dialogue_state,
                     inferred_topic, inferred_formality, inferred_tone],
        ).then(lambda: "", None, msg_box)

        set_persona_btn.click(set_persona, inputs=[persona_input], outputs=[state_persona])

        clear_btn.click(
            clear_conversation, outputs=[chatbot_ui, state_history, state_dialogue_state,
                                          inferred_topic, inferred_formality, inferred_tone],
        )

        feedback_btn.click(
            submit_feedback,
            inputs=[participant_code, session_id, state_history, rating, feedback_comment],
            outputs=[feedback_status],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_checkpoint", default=None,
                         help="Path to Stage-1 fine-tuned backbone (omit to use base DialoGPT-medium)")
    parser.add_argument("--stage2_checkpoint", default=None,
                         help="Path to Stage-2 checkpoint directory containing upe.pt / dst.pt")
    parser.add_argument("--share", action="store_true", help="Use Gradio's public URL sharing")
    parser.add_argument("--quick_demo", action="store_true", help="Run a quick demo without downloading models")
    parser.add_argument("--host", default=None, help="Host/IP for the Gradio server")
    parser.add_argument("--port", type=int, default=None, help="Port for the Gradio server")
    args = parser.parse_args()

    if args.quick_demo:
        system = StubPersonalisedChatbotSystem()
    else:
        system = build_system(args.backbone_checkpoint, args.stage2_checkpoint)
    demo = launch_app(system)

    env_port = int(os.environ.get("PORT", "0"))
    preferred_port = args.port if args.port is not None else (env_port if env_port > 0 else 0)
    candidate_ports = [preferred_port] if preferred_port > 0 else list(range(7860, 7875))

    if args.host:
        server_name = args.host
    elif env_port > 0:
        server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    else:
        server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")

    last_error = None
    for port in candidate_ports:
        try:
            print(f"Launching Gradio on http://{server_name}:{port}")
            demo.launch(
                share=args.share,
                server_name=server_name,
                server_port=port,
                quiet=(not getattr(args, "quick_demo", False)),
            )
            break
        except OSError as exc:
            last_error = exc
            print(f"Port {port} was unavailable; trying the next one.")
    else:
        raise last_error or OSError("Unable to launch Gradio on any candidate port")
