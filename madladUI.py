import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPlainTextEdit, QVBoxLayout, QPushButton, QMessageBox, QSpinBox
from PyQt5.QtCore import Qt
from transformers import T5ForConditionalGeneration, T5Tokenizer

class TextTranslationApp(QWidget):
    def __init__(self):
        super().__init__()

        self.model_name = 'jbochi/madlad400-3b-mt'
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, device_map="auto")
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)


        self.init_ui()

    def init_ui(self):

        author_label = QLabel('<a href="https://github.com/Minksh">My git </a>')
        author_label.setOpenExternalLinks(True)


        code_label = QLabel('<a href="https://huggingface.co/google/madlad400-3b-mt">This script requires Google\'s madlad400-3b-mt model </a>')
        code_label.setOpenExternalLinks(True)


        self.input_label = QLabel("Enter the text you want to translate:")
        self.input_text_widget = QPlainTextEdit(self)


        self.language_label = QLabel("Language Code:")
        self.language_entry = QPlainTextEdit(self)
        self.language_entry.setPlaceholderText("e.g., 2ar for Arabic, 2en for English")


        self.output_label = QLabel("Translation:")
        self.output_text_widget = QPlainTextEdit(self)
        self.output_text_widget.setLineWrapMode(QPlainTextEdit.WidgetWidth)  # Set line wrap mode
        self.output_text_widget.setReadOnly(True)


        self.max_new_tokens_label = QLabel("Max Tokens:")
        self.max_new_tokens_spinbox = QSpinBox(self)
        self.max_new_tokens_spinbox.setRange(1, 1000)
        self.max_new_tokens_spinbox.setValue(100)  # Default value


        self.translate_button = QPushButton("Translate", self)
        self.translate_button.clicked.connect(self.translate_text)


        self.copy_button = QPushButton("Copy Output Text", self)
        self.copy_button.clicked.connect(self.copy_output_text)


        layout = QVBoxLayout()
        layout.addWidget(author_label)
        layout.addWidget(code_label)
        layout.addWidget(self.input_label)
        layout.addWidget(self.input_text_widget)
        layout.addWidget(self.language_label)
        layout.addWidget(self.language_entry)
        layout.addWidget(self.output_label)
        layout.addWidget(self.output_text_widget)
        layout.addWidget(self.max_new_tokens_label)
        layout.addWidget(self.max_new_tokens_spinbox)
        layout.addWidget(self.translate_button)
        layout.addWidget(self.copy_button)


        self.setLayout(layout)


        self.setWindowTitle('madlad400-3b-mt-gui')
        self.setGeometry(100, 100, 600, 400)

    def translate_text(self):

        input_text = self.input_text_widget.toPlainText()
        language_code = self.language_entry.toPlainText()
        max_new_tokens = self.max_new_tokens_spinbox.value()


        text_with_token = f"<{language_code}> {input_text}"


        input_ids = self.tokenizer(text_with_token, return_tensors="pt").input_ids.to(self.model.device)


        outputs = self.model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)


        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.output_text_widget.setPlainText(translation)

    def copy_output_text(self):

        output_text = self.output_text_widget.toPlainText()
        QApplication.clipboard().setText(output_text)
        QMessageBox.information(self, 'Copied', 'Output text copied to clipboard.')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TextTranslationApp()
    window.show()
    sys.exit(app.exec_())