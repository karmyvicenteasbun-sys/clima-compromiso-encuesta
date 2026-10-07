"""
Genera los datos ficticios de la encuesta anual de clima de una embotelladora
mediana (520 colaboradores en 40 equipos), aplicada en septiembre de 2025, y las renuncias
voluntarias de cada equipo en los 12 meses siguientes (octubre de 2025 a
septiembre de 2026).

Salidas (carpeta datos/):
  - encuesta_clima_2025.csv  una fila por respuesta. Es anónima: no identifica a la persona.
  - preguntas.csv            código, dimensión, texto y sentido de cada pregunta
  - equipos.csv              una fila por equipo (jefatura directa): dotación y renuncias posteriores

Todo es inventado: no hay datos de ninguna empresa real.

Cómo se generan las respuestas:
  1. Cada persona tiene un nivel "latente" (no observado) en cada una de las 7
     dimensiones del clima. Esos niveles están correlacionados entre sí y dependen
     del área, el nivel, la antigüedad, el turno y la calidad del jefe directo.
  2. El compromiso latente depende sobre todo de Liderazgo y Reconocimiento.
  3. Cada ítem es: carga x latente + ruido, discretizado a la escala de 1 a 5.
  4. No todos responden: responden menos las áreas con más trabajo en la calle y
     las personas menos comprometidas.
  5. Las renuncias de cada equipo dependen del área, del nivel y la antigüedad de
     su gente y del compromiso latente de cada persona, con ruido.

Uso:
    python generar_datos.py
"""

from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

SEMILLA = 2026

# --- Equipos -----------------------------------------------------------------
# Un equipo = las personas que reportan a una misma jefatura directa.
# dotacion: personas del área en septiembre de 2025
# equipos: cantidad de jefaturas directas | rotativo: probabilidad de que el equipo rote turnos
AREAS = {
    "Producción":                dict(prefijo="PRO", dotacion=208, equipos=15, rotativo=0.65),
    "Logística y Distribución":  dict(prefijo="LOG", dotacion=125, equipos=10, rotativo=0.20),
    "Ventas":                    dict(prefijo="VEN", dotacion=73, equipos=6, rotativo=0.00),
    "Mantenimiento":             dict(prefijo="MAN", dotacion=42, equipos=3, rotativo=0.50),
    "Calidad":                   dict(prefijo="CAL", dotacion=31, equipos=2, rotativo=0.40),
    "Administración y Finanzas": dict(prefijo="ADM", dotacion=41, equipos=4, rotativo=0.00),
}
MINIMO_POR_EQUIPO = 5
DISPERSION_TAMANOS = 4   # menor = tamaños de equipo más desiguales dentro del área

NIVELES = ["Operativo", "Técnico y supervisión", "Profesional y jefatura"]
MEZCLA_NIVELES = {  # mismo orden que NIVELES
    "Producción":                [0.82, 0.13, 0.05],
    "Logística y Distribución":  [0.85, 0.11, 0.04],
    "Ventas":                    [0.82, 0.12, 0.06],
    "Mantenimiento":             [0.35, 0.55, 0.10],
    "Calidad":                   [0.45, 0.45, 0.10],
    "Administración y Finanzas": [0.40, 0.40, 0.20],
}

TRAMOS_ANTIGUEDAD = ["Menos de 1 año", "1 a 3 años", "3 a 5 años", "Más de 5 años"]
MEZCLA_ANTIGUEDAD = {  # mismo orden que TRAMOS_ANTIGUEDAD
    "Operativo":              [0.24, 0.30, 0.16, 0.30],
    "Técnico y supervisión":  [0.12, 0.25, 0.20, 0.43],
    "Profesional y jefatura": [0.10, 0.22, 0.22, 0.46],
}

# --- Dimensiones del clima ----------------------------------------------------
DIMENSIONES = ["Liderazgo del jefe directo", "Reconocimiento", "Desarrollo", "Carga de trabajo",
               "Compensación y beneficios", "Seguridad y condiciones", "Comunicación"]

# Correlación entre los latentes de una misma persona (mismo orden que DIMENSIONES)
CORRELACION_LATENTES = np.array([
    [1.00, 0.55, 0.45, 0.25, 0.25, 0.30, 0.50],
    [0.55, 1.00, 0.50, 0.25, 0.35, 0.25, 0.40],
    [0.45, 0.50, 1.00, 0.20, 0.35, 0.25, 0.40],
    [0.25, 0.25, 0.20, 1.00, 0.25, 0.30, 0.25],
    [0.25, 0.35, 0.35, 0.25, 1.00, 0.20, 0.25],
    [0.30, 0.25, 0.25, 0.30, 0.20, 1.00, 0.30],
    [0.50, 0.40, 0.40, 0.25, 0.25, 0.30, 1.00],
])

# Efectos sobre los latentes, en desvíos estándar (mismo orden que DIMENSIONES)
EFECTO_AREA = {
    "Producción":                [0.00, 0.00, -0.05, -0.15, 0.00, -0.10, 0.00],
    "Logística y Distribución":  [-0.75, -0.40, -0.25, -0.95, -0.15, -0.30, -0.30],
    "Ventas":                    [0.05, -0.05, 0.05, -0.20, -0.30, 0.15, -0.05],
    "Mantenimiento":             [0.10, 0.05, 0.20, 0.00, 0.05, 0.00, -0.10],
    "Calidad":                   [0.25, 0.15, 0.10, 0.10, 0.05, 0.25, 0.10],
    "Administración y Finanzas": [0.15, 0.10, -0.10, 0.25, 0.15, 0.40, 0.25],
}
EFECTO_NIVEL = {
    "Operativo":              [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    "Técnico y supervisión":  [0.05, 0.05, 0.15, -0.05, 0.10, 0.05, 0.10],
    "Profesional y jefatura": [0.10, 0.10, 0.30, -0.25, 0.25, 0.10, 0.30],
}
EFECTO_ANTIGUEDAD = {  # "luna de miel" del primer año y meseta en desarrollo después
    "Menos de 1 año": [0.15, 0.10, 0.20, 0.10, 0.05, 0.05, 0.10],
    "1 a 3 años":     [-0.05, -0.05, -0.05, -0.05, -0.10, 0.00, -0.05],
    "3 a 5 años":     [0.00, 0.00, -0.15, 0.00, -0.05, 0.00, 0.00],
    "Más de 5 años":  [0.00, 0.05, -0.10, 0.05, 0.05, 0.05, 0.05],
}
EFECTO_TURNO_ROTATIVO = [0.00, 0.00, 0.00, -0.25, 0.00, -0.10, -0.10]

# Calidad del jefe directo: un efecto por equipo, que pesa sobre todo en Liderazgo
DESVIO_JEFE = 0.65
PESO_JEFE = [1.00, 0.70, 0.35, 0.25, 0.00, 0.15, 0.45]

# --- Compromiso ---------------------------------------------------------------
# Peso de cada dimensión en el compromiso latente (mismo orden que DIMENSIONES)
PESOS_COMPROMISO = [0.38, 0.32, 0.17, 0.08, 0.15, 0.05, 0.06]
RUIDO_COMPROMISO = 0.55     # lo que el clima no explica (situación personal, mercado, etc.)
LUNA_DE_MIEL = 0.15         # compromiso extra del primer año

# --- De latente a respuesta -----------------------------------------------------
UMBRALES_LIKERT = [-1.75, -0.85, -0.15, 1.05]  # cortes que pasan el latente a la escala 1 a 5
ESTILO_RESPUESTA = 0.15     # tendencia personal a responder más alto o más bajo en todo
# Nivel medio de cada dimensión (desvíos): define cuán favorable sale cada una
MEDIA_DIMENSION = {
    "Liderazgo del jefe directo": 0.30, "Reconocimiento": -0.30, "Desarrollo": -0.05,
    "Carga de trabajo": 0.15, "Compensación y beneficios": -0.60, "Seguridad y condiciones": 0.50,
    "Comunicación": 0.00, "Compromiso": 0.10,
}

# código, dimensión, texto, carga en su latente, desplazamiento propio del ítem
# Carga negativa = ítem redactado en negativo (estar de acuerdo es desfavorable).
# Carga muy baja = ítem débil: mide otra cosa (un hábito, no una percepción).
PREGUNTAS = [
    ("lid_1", "Liderazgo del jefe directo", "Mi jefe directo me explica con claridad lo que espera de mí.", 0.80, 0.10),
    ("lid_2", "Liderazgo del jefe directo", "Mi jefe directo me trata con respeto.", 0.72, 0.45),
    ("lid_3", "Liderazgo del jefe directo", "Puedo plantearle un problema a mi jefe directo sin miedo a represalias.", 0.76, 0.00),
    ("lid_4", "Liderazgo del jefe directo", "Mi jefe directo me da retroalimentación útil sobre mi trabajo.", 0.78, -0.15),
    ("rec_1", "Reconocimiento", "Cuando hago un buen trabajo, alguien me lo reconoce.", 0.80, 0.00),
    ("rec_2", "Reconocimiento", "En los últimos tres meses recibí un reconocimiento por mi trabajo.", 0.68, -0.30),
    ("rec_3", "Reconocimiento", "Siento que mi aporte se valora en esta empresa.", 0.78, 0.10),
    ("rec_4", "Reconocimiento", "En mi área los reconocimientos se dan de forma justa.", 0.72, -0.05),
    ("des_1", "Desarrollo", "En el último año aprendí cosas nuevas en mi trabajo.", 0.62, 0.35),
    ("des_2", "Desarrollo", "Recibo la capacitación que necesito para hacer bien mi trabajo.", 0.74, 0.05),
    ("des_3", "Desarrollo", "Veo posibilidades reales de crecer dentro de la empresa.", 0.78, -0.25),
    ("des_4", "Desarrollo", "Las promociones internas se deciden con criterios claros.", 0.70, -0.20),
    ("car_1", "Carga de trabajo", "La cantidad de trabajo que tengo es razonable.", 0.80, 0.00),
    ("car_2", "Carga de trabajo", "Puedo terminar mi trabajo dentro de mi horario normal.", 0.74, -0.10),
    ("car_3", "Carga de trabajo", "Mi trabajo me deja sin energía al final del día.", -0.66, 0.10),
    ("car_4", "Carga de trabajo", "Tengo el tiempo necesario para hacer mi trabajo con calidad.", 0.72, 0.10),
    ("cyb_1", "Compensación y beneficios", "Mi sueldo es justo para el trabajo que hago.", 0.82, 0.00),
    ("cyb_2", "Compensación y beneficios", "Mi sueldo es competitivo frente a otras empresas de la zona.", 0.74, -0.15),
    ("cyb_3", "Compensación y beneficios", "Los beneficios de la empresa responden a mis necesidades.", 0.62, 0.15),
    ("cyb_4", "Compensación y beneficios", "Entiendo cómo se calcula lo que gano (bonos, horas extra, antigüedad).", 0.52, 0.45),
    ("seg_1", "Seguridad y condiciones", "Cuento con el equipo de protección que necesito.", 0.68, 0.20),
    ("seg_2", "Seguridad y condiciones", "En mi área se cumplen las normas de seguridad.", 0.74, 0.00),
    ("seg_3", "Seguridad y condiciones", "Mi lugar de trabajo tiene condiciones adecuadas (temperatura, ruido, limpieza).", 0.62, -0.35),
    ("seg_4", "Seguridad y condiciones", "Si reporto una condición insegura, se corrige.", 0.70, -0.05),
    ("com_1", "Comunicación", "Me entero a tiempo de los cambios que afectan mi trabajo.", 0.76, -0.05),
    ("com_2", "Comunicación", "La gerencia explica con claridad hacia dónde va la empresa.", 0.72, 0.00),
    ("com_3", "Comunicación", "La información entre áreas fluye bien.", 0.70, -0.25),
    ("com_4", "Comunicación", "Leo el boletín interno de la empresa.", 0.15, 0.10),
    ("cpr_1", "Compromiso", "Siento orgullo de trabajar en esta empresa.", 0.82, 0.15),
    ("cpr_2", "Compromiso", "Recomendaría esta empresa como un buen lugar para trabajar.", 0.84, 0.00),
    ("cpr_3", "Compromiso", "Me veo trabajando aquí dentro de dos años.", 0.72, 0.05),
    ("cpr_4", "Compromiso", "Doy lo mejor de mí en mi trabajo todos los días.", 0.58, 0.85),
]
TEXTO_ENPS = ("Del 0 al 10, ¿qué tan probable es que recomiende esta empresa como lugar de trabajo "
              "a un amigo o familiar?")
CARGA_ENPS = 0.80           # relación del eNPS con el compromiso latente
# Distribución acumulada del eNPS en toda la dotación (puntajes 0 a 9; el resto es 10)
ACUMULADA_ENPS = [0.03, 0.045, 0.06, 0.085, 0.115, 0.18, 0.27, 0.43, 0.65, 0.79]

# --- Participación ------------------------------------------------------------
PARTICIPACION = {  # probabilidad de responder para una persona de compromiso promedio
    "Producción": 0.86, "Logística y Distribución": 0.75, "Ventas": 0.82,
    "Mantenimiento": 0.90, "Calidad": 0.93, "Administración y Finanzas": 0.95,
}
EFECTO_COMPROMISO_EN_PARTICIPAR = 0.35   # log-odds por desvío de compromiso

# --- Renuncias voluntarias en los 12 meses siguientes (oct-2025 a sep-2026) ----
TASA_RENUNCIA_AREA = {  # tasa anual para compromiso promedio; Logística y Ventas compiten con el mercado
    "Producción": 0.11, "Logística y Distribución": 0.19, "Ventas": 0.18,
    "Mantenimiento": 0.07, "Calidad": 0.06, "Administración y Finanzas": 0.05,
}
FACTOR_RENUNCIA_NIVEL = {"Operativo": 1.00, "Técnico y supervisión": 0.80, "Profesional y jefatura": 0.65}
FACTOR_RENUNCIA_ANTIGUEDAD = {"Menos de 1 año": 1.80, "1 a 3 años": 1.10, "3 a 5 años": 0.70, "Más de 5 años": 0.50}
EFECTO_COMPROMISO_EN_RENUNCIA = 0.60     # un desvío más de compromiso: tasa x exp(-0,60) = x0,55
DISPERSION_EQUIPO = 15                   # forma de la gamma que agrega ruido propio de cada equipo


def armar_equipos(rng):
    filas = []
    for area, conf in AREAS.items():
        k = conf["equipos"]
        # se reparte la dotación del área entre sus equipos, con un mínimo por equipo
        pesos = rng.dirichlet(np.full(k, DISPERSION_TAMANOS))
        tamanos = MINIMO_POR_EQUIPO + rng.multinomial(conf["dotacion"] - MINIMO_POR_EQUIPO * k, pesos)
        for n, dotacion in enumerate(tamanos, start=1):
            turno = "Rotativo" if rng.random() < conf["rotativo"] else "Diurno"
            filas.append(dict(equipo=f"{conf['prefijo']}-{n:02d}", area=area, dotacion=int(dotacion), turno=turno,
                              efecto_jefe=rng.normal(0, DESVIO_JEFE)))
    return pd.DataFrame(filas)


def armar_personas(rng, equipos):
    filas = []
    for e in equipos.itertuples():
        for _ in range(e.dotacion):
            nivel = str(rng.choice(NIVELES, p=MEZCLA_NIVELES[e.area]))
            filas.append(dict(
                equipo=e.equipo, area=e.area, nivel=nivel,
                antiguedad=str(rng.choice(TRAMOS_ANTIGUEDAD, p=MEZCLA_ANTIGUEDAD[nivel])),
                # las jefaturas profesionales trabajan de día aunque su equipo rote
                turno=e.turno if nivel != "Profesional y jefatura" else "Diurno",
                efecto_jefe=e.efecto_jefe,
            ))
    return pd.DataFrame(filas)


def latentes(rng, personas):
    """Nivel latente de cada persona en las 7 dimensiones y en el compromiso."""
    medias = (np.array([EFECTO_AREA[a] for a in personas["area"]])
              + np.array([EFECTO_NIVEL[n] for n in personas["nivel"]])
              + np.array([EFECTO_ANTIGUEDAD[t] for t in personas["antiguedad"]])
              + np.outer(personas["turno"].eq("Rotativo"), EFECTO_TURNO_ROTATIVO)
              + np.outer(personas["efecto_jefe"], PESO_JEFE))
    ruido = rng.multivariate_normal(np.zeros(len(DIMENSIONES)), CORRELACION_LATENTES, size=len(personas))
    z = medias + ruido
    compromiso = (z @ np.array(PESOS_COMPROMISO)
                  + LUNA_DE_MIEL * personas["antiguedad"].eq("Menos de 1 año").to_numpy()
                  + rng.normal(0, RUIDO_COMPROMISO, len(personas)))
    compromiso = (compromiso - compromiso.mean()) / compromiso.std()
    return z, compromiso


def responder(rng, personas, z, compromiso):
    """Respuestas de 1 a 5 a cada ítem y eNPS de 0 a 10."""
    n = len(personas)
    estilo = rng.normal(0, ESTILO_RESPUESTA, n)
    respuestas = {}
    for codigo, dimension, _, carga, desplazamiento in PREGUNTAS:
        latente = compromiso if dimension == "Compromiso" else z[:, DIMENSIONES.index(dimension)]
        continuo = (MEDIA_DIMENSION[dimension] * np.sign(carga) + desplazamiento + carga * latente
                    + np.sqrt(1 - carga ** 2) * rng.normal(0, 1, n) + estilo)
        respuestas[codigo] = np.digitize(continuo, UMBRALES_LIKERT) + 1
    cortes_enps = [NormalDist().inv_cdf(p) for p in ACUMULADA_ENPS]
    enps = CARGA_ENPS * compromiso + np.sqrt(1 - CARGA_ENPS ** 2) * rng.normal(0, 1, n)
    respuestas["enps"] = np.digitize(enps, cortes_enps)
    return pd.DataFrame(respuestas, index=personas.index)


def renuncias_por_equipo(rng, personas, compromiso):
    tasa = (personas["area"].map(TASA_RENUNCIA_AREA)
            * personas["nivel"].map(FACTOR_RENUNCIA_NIVEL)
            * personas["antiguedad"].map(FACTOR_RENUNCIA_ANTIGUEDAD)
            * np.exp(-EFECTO_COMPROMISO_EN_RENUNCIA * compromiso))
    esperadas = tasa.groupby(personas["equipo"]).sum()
    ruido_equipo = rng.gamma(DISPERSION_EQUIPO, 1 / DISPERSION_EQUIPO, len(esperadas))
    return pd.Series(rng.poisson(esperadas.to_numpy() * ruido_equipo), index=esperadas.index)


def generar(semilla=SEMILLA):
    # Un generador aleatorio independiente para cada proceso: así, cambiar un
    # supuesto de un proceso no cambia los sorteos de los demás.
    rng_equipos, rng_personas, rng_latentes, rng_respuestas, rng_participacion, rng_renuncias = (
        np.random.default_rng(s) for s in np.random.SeedSequence(semilla).spawn(6))
    equipos = armar_equipos(rng_equipos)
    personas = armar_personas(rng_personas, equipos)
    z, compromiso = latentes(rng_latentes, personas)
    respuestas = responder(rng_respuestas, personas, z, compromiso)

    # Quién responde la encuesta
    base = np.log(personas["area"].map(PARTICIPACION) / (1 - personas["area"].map(PARTICIPACION)))
    prob_responder = 1 / (1 + np.exp(-(base + EFECTO_COMPROMISO_EN_PARTICIPAR * compromiso)))
    responde = rng_participacion.random(len(personas)) < prob_responder

    # Las filas se mezclan para que el orden no delate a nadie
    encuesta = pd.concat([personas[["equipo", "area", "nivel", "antiguedad", "turno"]], respuestas], axis=1)
    encuesta = encuesta[responde].sample(frac=1, random_state=semilla).reset_index(drop=True)

    equipos["renuncias_12m"] = equipos["equipo"].map(renuncias_por_equipo(rng_renuncias, personas, compromiso))
    equipos = equipos[["equipo", "area", "turno", "dotacion", "renuncias_12m"]]

    preguntas = pd.DataFrame(
        [(c, d, t, "inverso" if carga < 0 else "directo") for c, d, t, carga, _ in PREGUNTAS]
        + [("enps", "eNPS", TEXTO_ENPS, "escala 0 a 10")],
        columns=["codigo", "dimension", "texto", "sentido"])
    return encuesta, preguntas, equipos


def main():
    encuesta, preguntas, equipos = generar()
    carpeta = Path(__file__).resolve().parent / "datos"
    carpeta.mkdir(exist_ok=True)
    encuesta.to_csv(carpeta / "encuesta_clima_2025.csv", index=False, encoding="utf-8")
    preguntas.to_csv(carpeta / "preguntas.csv", index=False, encoding="utf-8")
    equipos.to_csv(carpeta / "equipos.csv", index=False, encoding="utf-8")
    dotacion = equipos["dotacion"].sum()
    print(f"encuesta_clima_2025.csv: {len(encuesta)} respuestas de {dotacion} colaboradores "
          f"({len(encuesta) / dotacion:.0%})")
    print(f"preguntas.csv: {len(preguntas)} preguntas")
    print(f"equipos.csv: {len(equipos)} equipos, {equipos['renuncias_12m'].sum()} renuncias en los 12 meses siguientes")


if __name__ == "__main__":
    main()
