# Author: Ali Amini |----> aliamini9728@gmail.com

import flet as ft
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os


# ===============================
# NOTE FOR DEVELOPERS:
# Instead of downloading NLTK data every time (stopwords, punkt, etc.),
# you can download them once and keep them in a local folder (e.g., ./nltk_data).
# Then, set the NLTK_DATA environment variable or use nltk.data.path.append("path").
#
# Later in your code:
#   import nltk
#   nltk.data.path.append('nltk_data')
#
# This way, new developers won’t need to redownload data on every machine.
# ===============================


error_message = None
try:
    model = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'spam_model.pkl'))
    vectorizer = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'vectorizer.pkl'))
except Exception as e:
    error_message = f"Error: {str(e)}"

try:
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
except Exception as download_err:
    error_message = f"Failed to download NLTK data: {str(download_err)}. Please connect to the internet and restart the app."
else:
    error_message = None


def preprocess_text(text):
    words = nltk.word_tokenize(text.lower())
    words = [stemmer.stem(word) for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

def main(page: ft.Page):
    page.title = "Spam Detector"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 20

    message_display = ft.Row(
        controls=[
            ft.Text(
                "Please enter your message" if not error_message else error_message,
                size=24,
                weight=ft.FontWeight.BOLD,
                text_align=ft.TextAlign.CENTER,
                color=ft.Colors.RED if error_message else None
            )
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        height=60
    )

    input_field = ft.TextField(
        label="Enter your message here",
        multiline=True,
        width=500,
        height=120,
        border_radius=10,
        min_lines=3,
        disabled=bool(error_message)
    )

    def toggle_theme():
        if error_message:
            return
        page.theme_mode = ft.ThemeMode.DARK if page.theme_mode == ft.ThemeMode.LIGHT else ft.ThemeMode.LIGHT
        page.update()

    def predict():
        if error_message:
            return
        message = input_field.value.strip()
        if not message:
            message_display.controls = [
                ft.Row([
                    ft.Icon(ft.Icons.WARNING, color=ft.Colors.ORANGE, size=24),
                    ft.Text(
                        "Please enter a message!",
                        size=20,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.ORANGE
                    )
                ], alignment=ft.MainAxisAlignment.CENTER)
            ]
        else:
            try:
                processed = preprocess_text(message)
                vectorized = vectorizer.transform([processed])
                prediction = model.predict(vectorized)[0]
                if prediction == 1:
                    message_display.controls = [
                        ft.Row([
                            ft.Icon(ft.Icons.DANGEROUS, color=ft.Colors.RED, size=28),
                            ft.Text(
                                "SPAM",
                                size=24,
                                weight=ft.FontWeight.BOLD,
                                color=ft.Colors.RED
                            )
                        ], alignment=ft.MainAxisAlignment.CENTER)
                    ]
                else:
                    message_display.controls = [
                        ft.Row([
                            ft.Icon(ft.Icons.CHECK_CIRCLE, color=ft.Colors.GREEN, size=28),
                            ft.Text(
                                "HAM",
                                size=24,
                                weight=ft.FontWeight.BOLD,
                                color=ft.Colors.GREEN
                            )
                        ], alignment=ft.MainAxisAlignment.CENTER)
                    ]
            except Exception as e:
                message_display.controls = [
                    ft.Row([
                        ft.Icon(ft.Icons.ERROR, color=ft.Colors.RED, size=24),
                        ft.Text(
                            f"Prediction error: {str(e)}",
                            size=24,
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.RED
                        )
                    ], alignment=ft.MainAxisAlignment.CENTER)
                ]
        page.update()

    buttons_row = ft.Row(
        controls=[
            ft.ElevatedButton(
                "Change Theme",
                icon=ft.Icons.BRIGHTNESS_6,
                on_click=lambda _: toggle_theme(),
                width=150,
                height=45,
                disabled=bool(error_message)
            ),
            ft.ElevatedButton(
                "Check",
                icon=ft.Icons.SEARCH,
                on_click=lambda _: predict(),
                width=150,
                height=45,
                bgcolor=ft.Colors.BLUE,
                color=ft.Colors.WHITE,
                disabled=bool(error_message)
            )
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=20
    )

    main_column = ft.Column(
        controls=[
            message_display,
            ft.Container(height=20),
            input_field,
            ft.Container(height=20),
            buttons_row
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=0
    )


    footer = ft.Container(
        content=ft.Text(
            "Developed by Ali Amini | Powered by Flet",
            color=ft.Colors.BLUE_100,
            size=12,
            text_align=ft.TextAlign.CENTER,
            weight=ft.FontWeight.BOLD
        ),
        bgcolor=ft.Colors.BLUE_GREY_900,
        padding=6,
        bottom=0,
        left=0,
        right=0,
        alignment=ft.alignment.bottom_center,
        border=ft.border.only(top=ft.border.BorderSide(1, ft.Colors.BLUE_GREY))
    )

    page.add(main_column)
    page.overlay.append(footer)
    page.add(main_column)

ft.app(target=main)