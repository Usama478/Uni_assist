from __future__ import annotations
import gradio as gr
from engine.chat_handler import generate_response

def handle_user_message(user_input: str, chat_state: list):
    chat_state = chat_state or []
    history_tuples = [
        (chat_state[i]["content"], chat_state[i + 1]["content"])
        for i in range(0, len(chat_state) - 1, 2)
        if chat_state[i]["role"] == "user" and chat_state[i + 1]["role"] == "assistant"
    ]
    ai_response = generate_response(user_query=user_input, conversation_history=history_tuples)
    chat_state.append({"role": "user", "content": user_input})
    chat_state.append({"role": "assistant", "content": ai_response})
    return "", chat_state

# CSS styling
css_style = """
.gradio-container {
    max-width: 100% !important;
    background: #dfe6f1 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

#main-container {
    display: flex;
    flex-direction: column;
    height: 100vh;
}

#title-bar {
    background: #2f4c8a;
    padding: 16px 0;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
}

#title-bar h1 {
    margin: 0;
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    text-align: center;
}

#chat-area {
    flex: 1;
    overflow: hidden;
    padding: 20px;
    background: #dfe6f1;
}

#chatbot {
    height: 100% !important;
    border: 2px solid #2f4c8a !important;
    background: #ffffff !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

#chatbot .user {
    background: #e6f2ff !important;
    color: #fff !important;
    align-self: flex-end !important;
    border-radius: 12px !important;
}

#chatbot .bot {
    background: #ffffff !important;
    border: 2px solid #2f4c8a !important;
    color: #000 !important;
    align-self: flex-start !important;
    border-radius: 12px !important;
}

#input-area {
    background: transparent !important;
    padding: 0 !important;
    border: none !important;
}

#input-row {
    display: flex;
    max-width: 800px;
    margin: 0 auto;
    padding: 16px 0;
    align-items: center;
    justify-content: center;
    position: relative;
}

#input-box textarea {
    background: #e6f2ff !important; /* soft light blue */
    border: 2px solid #2f4c8a !important;
    border-radius: 24px !important;
    padding: 12px 48px 12px 16px !important;
    font-size: 1rem !important;
    color: #000000 !important;
    width: 100% !important;
    box-sizing: border-box;
}


#send-btn {
    position: absolute;
    right: 16px;
    background: none !important;
    border: none !important;
    color: #0057ff !important;
    font-size: 1.4rem !important;
    cursor: pointer;
    padding: 4px;
    transition: color 0.3s ease, text-shadow 0.3s ease;
}

#send-btn:hover {
    color: #003dcc !important;
    text-shadow: 0 0 6px rgba(0, 87, 255, 0.5);
}
"""

# Build the Gradio interface
with gr.Blocks(fill_height=True) as app:
    with gr.Column(elem_id="main-container"):
        with gr.Row(elem_id="title-bar"):
            gr.Markdown("<h1>UMT Assist</h1>")
        
        with gr.Row(elem_id="chat-area"):
            chat_display = gr.Chatbot(
                elem_id="chatbot",
                show_label=False,
                container=False,
            )
        
        with gr.Row(elem_id="input-area"):
            with gr.Row(elem_id="input-row"):
                text_input = gr.Textbox(
                    placeholder="Type your message...",
                    show_label=False,
                    scale=8,
                    container=False,
                    elem_id="input-box"
                )
                submit_btn = gr.Button("➤", elem_id="send-btn", scale=0)

    text_input.submit(handle_user_message, [text_input, chat_display], [text_input, chat_display])
    submit_btn.click(handle_user_message, [text_input, chat_display], [text_input, chat_display])

if __name__ == "__main__":
    app.launch(css=css_style, theme=gr.themes.Soft(primary_hue="slate", neutral_hue="slate"))
