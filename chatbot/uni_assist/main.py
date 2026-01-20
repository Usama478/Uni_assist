"""
UNi Assist - Main Application
=============================================
Web interface for the AI-powered academic assistant.
"""

from __future__ import annotations

import gradio as gr

from engine.chat_handler import generate_response


def handle_user_message(user_input: str, chat_state: list):
    """Process incoming user messages and generate AI responses."""
    chat_state = chat_state or []
    
    # Transform Gradio's message format into conversation history tuples
    history_tuples = [
        (chat_state[i]["content"], chat_state[i + 1]["content"])
        for i in range(0, len(chat_state) - 1, 2)
        if chat_state[i]["role"] == "user" and chat_state[i + 1]["role"] == "assistant"
    ]
    
    ai_response = generate_response(user_query=user_input, conversation_history=history_tuples)
    
    # Append new messages in Gradio's expected format
    chat_state.append({"role": "user", "content": user_input})
    chat_state.append({"role": "assistant", "content": ai_response})
    
    return "", chat_state


interface_styles = """
.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    background: linear-gradient(135deg, #001a4d 0%, #003366 50%, #004d99 100%) !important;
}

#main-container {
    display: flex;
    flex-direction: column;
    height: 100vh;
    max-height: 100vh;
}

#title-bar {
    background: linear-gradient(90deg, #001a4d 0%, #003366 100%);
    padding: 24px 40px;
    text-align: center;
    border-bottom: 2px solid #FFFFFF;
    box-shadow: 0 4px 20px rgba(255, 255, 255, 0.2);
}

#title-bar h1 {
    margin: 0;
    font-size: 2rem;
    font-weight: 700;
    font-family: 'Inter', 'Segoe UI', 'Roboto', sans-serif;
    letter-spacing: 0.02em;
    color: #FFFFFF;
}

#chat-area {
    flex: 1;
    overflow: hidden;
    padding: 24px 40px;
    background: transparent;
}

#chatbot {
    height: 100% !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 16px !important;
    background: rgba(0, 26, 77, 0.8) !important;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
}

#chatbot .message {
    border-radius: 12px !important;
    padding: 14px 18px !important;
    margin: 8px 12px !important;
}

#chatbot .user {
    background: linear-gradient(135deg, #0052cc 0%, #0066ff 100%) !important;
    color: #FFFFFF !important;
    border-bottom-right-radius: 4px !important;
}

#chatbot .bot {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    color: #FFFFFF !important;
    border-bottom-left-radius: 4px !important;
}

#input-area {
    background: linear-gradient(90deg, #001a4d 0%, #003366 100%);
    padding: 20px 40px 24px;
    border-top: 1px solid rgba(255, 255, 255, 0.3);
}

#input-row {
    display: flex;
    gap: 12px;
    max-width: 1400px;
    margin: 0 auto;
}

#input-box textarea {
    border: 2px solid rgba(255, 255, 255, 0.4) !important;
    border-radius: 12px !important;
    padding: 14px 16px !important;
    font-size: 1rem !important;
    background: rgba(0, 52, 102, 0.6) !important;
    color: #FFFFFF !important;
    transition: all 0.3s ease !important;
}

#input-box textarea:focus {
    border-color: #FFFFFF !important;
    box-shadow: 0 0 20px rgba(255, 255, 255, 0.3) !important;
    outline: none !important;
}

#input-box textarea::placeholder {
    color: rgba(255, 255, 255, 0.5) !important;
}

#send-btn, #clear-btn {
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    min-width: 110px !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

#send-btn {
    background: linear-gradient(135deg, #0052cc 0%, #0066ff 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(0, 82, 204, 0.4) !important;
}

#send-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(0, 82, 204, 0.5) !important;
}

#clear-btn {
    background: transparent !important;
    border: 2px solid rgba(255, 255, 255, 0.5) !important;
    color: #FFFFFF !important;
}

#clear-btn:hover {
    background: rgba(255, 255, 255, 0.15) !important;
    border-color: #FFFFFF !important;
}
"""

# Build the Gradio interface
with gr.Blocks(fill_height=True) as application:
    with gr.Column(elem_id="main-container"):
        with gr.Row(elem_id="title-bar"):
            gr.Markdown("# Uni Assist")
        
        with gr.Row(elem_id="chat-area"):
            chat_display = gr.Chatbot(
                elem_id="chatbot",
                show_label=False,
                container=False,
            )
        
        with gr.Row(elem_id="input-area"):
            with gr.Row(elem_id="input-row"):
                text_input = gr.Textbox(
                    placeholder="Enter your question here...",
                    show_label=False,
                    scale=8,
                    container=False,
                    elem_id="input-box"
                )
                submit_btn = gr.Button("Send", elem_id="send-btn", scale=1)
                reset_btn = gr.Button("Clear", elem_id="clear-btn", scale=1)

    text_input.submit(handle_user_message, [text_input, chat_display], [text_input, chat_display])
    submit_btn.click(handle_user_message, [text_input, chat_display], [text_input, chat_display])
    reset_btn.click(lambda: [], None, chat_display)

if __name__ == "__main__":
    application.launch(css=interface_styles, theme=gr.themes.Soft(primary_hue="slate", neutral_hue="slate"))
