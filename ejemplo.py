import requests
from bs4 import BeautifulSoup
import pandas as pd
import spacy
from collections import Counter
import re
import sys
from datetime import datetime

# CONFIGURACIÓN

print("--- Iniciando Script de Análisis NLP: Cristiano Ronaldo ---")

# Cargar modelo de inglés de spaCy
try:
    nlp_en = spacy.load("en_core_web_sm")
    print("✅ Modelo NLP Inglés cargado correctamente.")
except OSError:
    print("❌ Error: No se encontró el modelo 'en_core_web_sm'")
    print("👉 Ejecuta en tu terminal: python -m spacy download en_core_web_sm")
    sys.exit(1)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; EstudianteSistemas/1.0; +http://tusitio.com)"
}

# 1. FUNCIÓN DE SCRAPING

def scrapear_wikipedia(url):
    """
    Descarga el contenido, extrae título y limpia el texto principal.
    """
    print(f"📡 Conectando a: {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Wikipedia usa 'firstHeading' para el título
        titulo_tag = soup.find("h1", id="firstHeading")
        titulo = titulo_tag.text.strip() if titulo_tag else "Sin Título"

        # El contenido principal suele estar en 'mw-content-text' o 'bodyContent'
        contenido_div = soup.find("div", id="bodyContent")
        if not contenido_div:
            return titulo, ""

        # Extraer solo párrafos para evitar tablas y menús laterales
        parrafos = contenido_div.find_all("p")
        texto_sucio = "\n".join(p.get_text() for p in parrafos)

        # --- LIMPIEZA DE DATOS (REGEX) ---
        # 1. Eliminar referencias tipo [1], [nota 2]
        texto = re.sub(r"\[.*?\]", "", texto_sucio)
        # 2. Eliminar paréntesis vacíos o con contenido muy corto irrelevante si se desea
        # texto = re.sub(r"\(.*?\)", "", texto) (Opcional: a veces elimina info útil)
        # 3. Eliminar espacios múltiples y saltos de línea excesivos
        texto = re.sub(r"\s+", " ", texto).strip()

        return titulo, texto

    except Exception as e:
        print(f"❌ Error al scrapear {url}: {e}")
        return None, None

# ==========================================
# 2. PROCESAMIENTO NLP (INGLÉS)
# ==========================================

def analizar_texto_ingles(texto, top_n=15):
    """
    Usa spaCy para lematizar y distinguir POS (Part of Speech).
    """
    print("⚙️ Procesando NLP Inglés...")
    # Aumentamos el límite de longitud por si el artículo es muy largo
    nlp_en.max_length = 2000000 
    doc = nlp_en(texto)

    palabras = []
    verbos = []

    for token in doc:
        # Filtros: que sea alfabético y no sea stopword
        if token.is_alpha and not token.is_stop and len(token.text) > 2:
            lemma = token.lemma_.lower()
            
            if token.pos_ == "VERB":
                verbos.append(lemma)
            elif token.pos_ in ["NOUN", "PROPN", "ADJ"]: 
                # Opcional: Centrarse en Sustantivos/Adjetivos para "palabras"
                palabras.append(lemma)

    return (
        Counter(palabras).most_common(top_n),
        Counter(verbos).most_common(top_n)
    )

# ==========================================
# 3. PROCESAMIENTO NLP (LATÍN)
# ==========================================

def analizar_texto_latin_basico(texto, top_n=15):
    """
    Análisis heurístico para Latín (sin modelo ML complejo).
    """
    print("⚙️ Procesando NLP Latín (Heurístico)...")

    # Lista ampliada de stopwords comunes en latín
    stopwords_latin = {
        'et', 'in', 'de', 'ad', 'non', 'ut', 'cum', 'per', 'sed',
        'qui', 'quae', 'quod', 'est', 'sunt', 'fuit', 'ex', 'ab',
        'se', 'is', 'ea', 'id', 'ac', 'atque', 'aut', 'autem',
        'enim', 'etiam', 'ibi', 'iam', 'ita', 'nam', 'ne', 'nec',
        'nunc', 'quam', 'quia', 'sic', 'tam', 'tamen', 'ubi', 'vel'
    }

    # Tokenización simple por regex
    palabras_raw = re.findall(r'\b[a-zA-Z]+\b', texto.lower())

    palabras_filtradas = [
        p for p in palabras_raw
        if p not in stopwords_latin and len(p) > 2
    ]

    # Heurística de verbos (Sufijos comunes de infinitivos y conjugaciones)
    # Nota: Esto es una aproximación, 'mare' (mar) termina en 'are' pero no es verbo.
    terminaciones_verbos = (
        'are', 'ere', 'ire', 'sse',        # Infinitivos
        'at', 'et', 'it', 'ant', 'ent', 'unt', 'bat', 'bant', # 3ra persona
        'avit', 'evit', 'ivit'             # Pretérito perfecto
    )

    lista_verbos = []
    lista_palabras = []

    for p in palabras_filtradas:
        if p.endswith(terminaciones_verbos):
            lista_verbos.append(p)
        else:
            lista_palabras.append(p)

    return (
        Counter(lista_palabras).most_common(top_n),
        Counter(lista_verbos).most_common(top_n)
    )

# ==========================================
# 4. EJECUCIÓN PRINCIPAL
# ==========================================

def main():
    # URLs
    url_en = "https://en.wikipedia.org/wiki/Cristiano_Ronaldo"
    url_la = "https://la.wikipedia.org/wiki/Christianus_Ronaldo"

    # 1. Scraping
    titulo_en, texto_en = scrapear_wikipedia(url_en)
    titulo_la, texto_la = scrapear_wikipedia(url_la)

    if not texto_en or not texto_la:
        print("❌ No se pudo completar el scraping. Revisa tu conexión.")
        return

    # 2. Análisis
    en_words, en_verbs = analizar_texto_ingles(texto_en)
    la_words, la_verbs = analizar_texto_latin_basico(texto_la)

    # 3. Exportación a Excel (Un solo archivo, múltiples pestañas)
    fecha = datetime.now().strftime("%Y%m%d_%H%M")
    nombre_archivo = f"Reporte_CR7_{fecha}.xlsx"

    print(f"💾 Guardando datos en {nombre_archivo}...")

    try:
        with pd.ExcelWriter(nombre_archivo, engine='openpyxl') as writer:
            # Hoja 1: Datos Crudos (Texto completo)
            df_raw = pd.DataFrame({
                "Idioma": ["Inglés", "Latín"],
                "Título": [titulo_en, titulo_la],
                "Contenido_Limpio": [texto_en[:30000], texto_la[:30000]] # Limitamos caracteres por celda de excel
            })
            df_raw.to_excel(writer, sheet_name="Texto_Original", index=False)

            # Hoja 2: Inglés - Frecuencias
            df_en_w = pd.DataFrame(en_words, columns=["Palabra (EN)", "Frecuencia"])
            df_en_v = pd.DataFrame(en_verbs, columns=["Verbo (EN)", "Frecuencia"])
            
            # Ponemos palabras y verbos lado a lado en la misma hoja
            df_en_final = pd.concat([df_en_w, df_en_v], axis=1)
            df_en_final.to_excel(writer, sheet_name="Analisis_Ingles", index=False)

            # Hoja 3: Latín - Frecuencias
            df_la_w = pd.DataFrame(la_words, columns=["Palabra (LA)", "Frecuencia"])
            df_la_v = pd.DataFrame(la_verbs, columns=["Verbo (LA)", "Frecuencia"])
            
            df_la_final = pd.concat([df_la_w, df_la_v], axis=1)
            df_la_final.to_excel(writer, sheet_name="Analisis_Latin", index=False)

        print("✅ ¡Proceso completado con éxito!")
        print(f"   Revisa el archivo: {nombre_archivo}")

    except Exception as e:
        print(f"❌ Error al guardar Excel: {e}")

if __name__ == "__main__":
    main()