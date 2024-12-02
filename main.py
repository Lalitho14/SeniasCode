import flet as ft
import cv2
from PIL import Image
import base64
from io import BytesIO
import threading
import mediapipe as mp
from mediapipe.python.solutions.holistic import Holistic
from tools_mediapipe import MediapipeDetection
from procesar_imagen import ProcesarImagen
from inference_rest import PrediccionLetraServer


def main(page: ft.Page):
  camara = ft.Image()
  letra = ft.Text("Letra detectada: ", size=30, weight="bold")

  def CaputurarCamara():
    margin_frame = 5
    count_frame = 0

    with Holistic() as holistic_model:
      cap = cv2.VideoCapture(0)

      while True:
        ret, frame = cap.read()

        if not ret:
          break

        frame = cv2.flip(frame, 1)

        results = MediapipeDetection(frame, holistic_model)

        if (results.left_hand_landmarks or results.right_hand_landmarks):
          count_frame += 1
          if count_frame > margin_frame:
            captura = cv2.resize(frame, (1280, 720))
            cv2.imwrite(f"senias/test.jpg", captura,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            ProcesarImagen("senias/test.jpg")
            print("Imagen tomada")
            letra.value = "Letra detectada: " + PrediccionLetraServer()
            count_frame = 0

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        img_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        camara.src_base64 = img_str
        page.update()

      cap.release()

  thread = threading.Thread(target=CaputurarCamara)
  thread.start()

  page.add(
    ft.Row(
      controls=[
          ft.Column(
              [
                  ft.Text(
                      "prueba de camera",
                      size=30,
                      weight="bold"
                  ),
                  camara,
                  letra
              ],
              alignment=ft.MainAxisAlignment.CENTER,
              horizontal_alignment=ft.CrossAxisAlignment.CENTER,
              expand=True
          )
      ],
      alignment=ft.MainAxisAlignment.CENTER,
      expand=True,
    )
  )


ft.app(target=main, assets_dir="senias")
