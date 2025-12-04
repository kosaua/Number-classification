import pickle
import random
import time
import io
import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = "https://www.googleapis.com/auth/drive.readonly"

def get_drive():
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    else:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('drive', 'v3', credentials=creds)


def download_data(folder_id, target="downloaded"):
    service = get_drive()

    # Pobieramy listę plików z Google Drive
    results = service.files().list(
        q=f"'{folder_id}' in parents", 
        fields="files(id, name, mimeType, size)", 
        pageSize=1000
    ).execute()

    files = results.get('files', [])
    downloaded = 0
    skipped = 0

    os.makedirs(target, exist_ok=True)
    
    print(f"Sprawdzam {len(files)} plików w chmurze...")

    for f in files:
        # Pomiń foldery
        if f.get('mimeType') == 'application/vnd.google-apps.folder':
            continue

        file_path = os.path.join(target, f['name'])

        # --- POPRAWKA: Sprawdź czy plik już istnieje ---
        if os.path.exists(file_path):
            # Opcjonalnie: można sprawdzać też rozmiar pliku, ale dla prostoty sprawdzamy tylko istnienie
            skipped += 1
            continue
        # -----------------------------------------------

        # Jeśli nie istnieje, pobierz go
        try:
            request = service.files().get_media(fileId=f['id'])
            with io.FileIO(file_path, 'wb') as file:
                downloader = MediaIoBaseDownload(file, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
            
            downloaded += 1
            # Wyświetl kropkę dla każdego pobranego pliku, żeby widzieć postęp
            print(".", end="", flush=True) 
            
        except Exception as e:
            print(f"\nBłąd pobierania {f['name']}: {e}")

    print(f"\nZakończono. Pobrano: {downloaded}, Pominięto (już istniejące): {skipped}")
    return target
