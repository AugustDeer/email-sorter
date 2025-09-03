from copy import replace

import gradio as gr

from . import DEFAULT_EMAIL_FILE, Email, loadEmails, saveEmails

with gr.Blocks() as demo:
    emails = gr.State([])
    idx_state = gr.State(0)

    with gr.Column() as email_download_col:
        dest = gr.Textbox(DEFAULT_EMAIL_FILE, label="Emails Location")
        with gr.Row(equal_height=True):
            load_btn = gr.Button("Load Emails")
            save_btn = gr.Button("Save Emails")

    def load_emails(path, progress=gr.Progress()):
        with open(path, "rb") as file:
            emails = loadEmails(file)
        return emails

    load_btn.click(load_emails, inputs=dest, outputs=emails)

    def save_emails(emails: list[Email], path: str):
        if len(emails) == 0:
            gr.Warning("No emails loaded.")
            return
        with open(path, "wb") as file:
            saveEmails(emails, file)

    save_btn.click(save_emails, inputs=[emails, dest])

    @gr.render(inputs=[emails, idx_state])
    def render_emails(emails: list[Email], idx: int):
        if len(emails) == 0:
            gr.Markdown("## No Emails Loaded")
        else:
            idx_box = gr.Slider(0, len(emails), idx, label="Email #")
            idx_box.input(lambda x: x, inputs=idx_box, outputs=idx_state)

            email = emails[idx]

            gr.Markdown(f"### *Subject*: {email.subject}")
            with gr.Accordion("Body", open=False, key="body"):
                gr.HTML(email.body)
            with gr.Row():
                spam_btn = gr.Button("Spam")
                ham_btn = gr.Button("Ham")

            def label(idx, label):
                emails[idx] = replace(email, spam=label)
                return idx + 1

            spam_btn.click(
                lambda idx: label(idx, True), inputs=idx_state, outputs=idx_state
            )
            ham_btn.click(
                lambda idx: label(idx, False), inputs=idx_state, outputs=idx_state
            )


if __name__ == "__main__":
    demo.launch()
