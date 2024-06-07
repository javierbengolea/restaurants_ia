import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Ex-stream-ly Cool ML",
    layout="wide")

style_css = '''
<style>div.block-container{padding-top: 0rem}</style>
'''
style_css_2 = '''
<style>
[data-testid="stAppViewContainer"]{}
[data-testid="stHeader"]{background-color: rgb(0,0,0,0)}
[data-testid="stToolbar"]{right: 2rem}
[data-testid="stForm"]{background-color: #fafafa; width: 80%; overflow-x: auto}


</style>'''

st.markdown(style_css, unsafe_allow_html=True)
st.markdown(style_css_2     , unsafe_allow_html=True)
st.title("Restaurantes y Características")


# Datos proporcionados
@st.cache_data
def read_data():
    X_subway = pd.read_parquet("ML/X_subway.parquet")
    X_subway_proc = pd.read_parquet("ML/X_subway_proc.parquet")
    filename = 'ML/finalized_model.pickle'
    # X_subway = pd.read_parquet("ML/X_.parquet")
    # X_subway_proc = pd.read_parquet("ML/X_proc.parquet")
    # filename = 'ML/modelo_89.pickle'
    # columnas_ignorar = ['index']
    # columnas_ignorar = X_subway_proc.iloc[:,69:].columns
    # def calcular_puntaje(df, columnas_ignorar):
    #     """
    #     Calcula el puntaje de los restaurantes y agrega una columna nueva al dataframe con el puntaje.

    #     Args:
    #     df (pd.DataFrame): DataFrame de restaurantes.
    #     columnas_ignorar (list): Lista de nombres de columnas que no se sumarán para calcular el puntaje base.

    #     Returns:
    #     pd.DataFrame: DataFrame con una columna nueva 'puntaje'.
    #     """
    #     # Identificar las columnas de cada grupo
    #     grupos = {
    #         'service': [col for col in df.columns if col.startswith('service')],
    #         'access': [col for col in df.columns if col.startswith('access')],
    #         'amen': [col for col in df.columns if col.startswith('amen')],
    #         'atmos': [col for col in df.columns if col.startswith('atmos')],
    #         'crowd': [col for col in df.columns if col.startswith('crowd')],
    #         'dining': [col for col in df.columns if col.startswith('dining')],
    #         'health': [col for col in df.columns if col.startswith('health')],
    #         'high': [col for col in df.columns if col.startswith('high')],
    #         'offer': [col for col in df.columns if col.startswith('offer')],
    #         'pay': [col for col in df.columns if col.startswith('pay')],
    #         'popular': [col for col in df.columns if col.startswith('popular')],
    #     }

    #     # Identificar todas las columnas que no están en la lista de columnas a ignorar
    #     columnas_suma = [col for col in df.columns if col not in columnas_ignorar]

    #     # Calcular el puntaje base sumando los valores de las columnas especificadas
    #     df['puntaje'] = df[columnas_suma].sum(axis=1)

    #     # Sumar un punto extra por cada grupo que tiene más de un valor presente
    #     for nombre_grupo, columnas_grupo in grupos.items():
    #         df['extra'] = df[columnas_grupo].apply(lambda row: row.sum() > 1, axis=1).astype(int)
    #         df['puntaje'] += df['extra']

    #     # Eliminar la columna 'extra' utilizada para el cálculo
    #     df.drop(columns=['extra'], inplace=True)

    #     return df

    # X_subway_proc = calcular_puntaje(X_subway_proc, columnas_ignorar)
    modelo = pickle.load(open(filename, 'rb'))
    return X_subway, X_subway_proc, modelo

X_subway, X_subway_proc, modelo = read_data()

# st.dataframe(X_subway)
# st.dataframe(X_subway.sample(1).T)


def get_atributos(id_restaurante):
    
    df = X_subway.query(f"id_restaurante == '{id_restaurante}'")
    tipo_atributo = [a.split("_")[0] for a in df.columns.tolist()[1:70]]
    atributo = [a.split("_")[1] for a in df.columns.tolist()[1:70]]
    atributos_dict = {'access': {}, 'amen': {}, 'atmos': {},  'crowd': {},  'dining': {},  'health': {},
     'high': {},  'offer': {},  'pay': {}, 'popular': {},  'service': {}}
   
    for ta, a in zip(tipo_atributo, atributo):
            atributos_dict[ta][a] = df.loc[:,f"{ta}_{a}"].values[0]

            
    return atributos_dict

try:
    muestra = X_subway.query(f"id_restaurante == '{st.query_params['id']}'")
except:
    muestra = X_subway.query(f"id_restaurante == '0x87a71b04e42e4c6d:0xdc11ec9338940205'")



# st.write(muestra.drop(columns=['index']))

# st.write(muestra['id_restaurante'].iloc[0])

# st.dataframe(X_subway_proc.loc[muestra.index])
calificacion = {0: "😢 Mala", 1: "😀 Buena"}

calificacion_restaurant = calificacion[modelo.predict(X_subway_proc.loc[muestra.index])[0]]

# Función para actualizar el valor
def update_calificacion(valor):
    st.session_state['calificacion_restaurant'] = valor

# Inicializar el valor en el estado de la sesión
if 'calificacion_restaurant' not in st.session_state:
    st.session_state['calificacion_restaurant'] = calificacion_restaurant

# Mostrar el valor actual
# st.write(f"Calificacion Inicial: {st.session_state['calificacion_restaurant']}")


# pprint(get_atributos("0x865681564f2dfd47:0x1f030438f1ceed23"))
data_2 = get_atributos(muestra['id_restaurante'].iloc[0])

nombres_atributo = {'access': 'Accesibility', 'amen': 'Amenities', 'atmos': 'Atmosphere', 'crowd':'Crowd',
'dining':'Dining Options','health':'Health and Safety','high':'Highlights','offer':'Offering','pay':'Payment',
'popular':'Popular for', 'service':'Services'}

# st.write(data_2)

with st.form("atributos_form"):
    submited_data = {}
    t_access, t_amen, t_atmos, t_crowd, t_dining, t_health, t_high, t_offer, t_pay, t_popular, t_service = st.tabs([i for i in nombres_atributo.values()])

    with t_access:
        c_access = st.container()
        
        with c_access:
            prefix = 'access'
            col1, col2 = st.columns([1,3])

            with col1:
                nombres_atributo[prefix]
            with col2:
                for tipo_atributo in data_2[prefix].items():
                    check_key = f"{prefix}_{tipo_atributo[0]}"
                    is_checked = col2.checkbox(tipo_atributo[0], key=check_key, value=bool(data_2[prefix][tipo_atributo[0]]))
                    submited_data[check_key] = int(is_checked)

    with t_amen:
        c_amen = st.container()

        with c_amen:
            prefix = 'amen'
            col1, col2 = st.columns([1,3])

            with col1:
                nombres_atributo['amen']
            with col2:
                for tipo_atributo in data_2[prefix].items():
                    check_key = f"{prefix}_{tipo_atributo[0]}"
                    is_checked = col2.checkbox(tipo_atributo[0], key=check_key, value=bool(data_2[prefix][tipo_atributo[0]]))
                    submited_data[check_key] = int(is_checked)

    with t_atmos:
        c_atmos = st.container()
        with c_atmos:
            prefix = 'atmos'
            col1, col2 = st.columns([1,3])

            with col1:
                nombres_atributo[prefix]
            with col2:
                for tipo_atributo in data_2[prefix].items():
                    check_key = f"{prefix}_{tipo_atributo[0]}"
                    is_checked = col2.checkbox(tipo_atributo[0], key=check_key, value=bool(data_2[prefix][tipo_atributo[0]]))
                    submited_data[check_key] = int(is_checked)
    with t_crowd:
        c_crowd = st.container()
        with c_crowd:
            prefix = 'crowd'
            col1, col2 = st.columns([1,3])
            with col1:
                nombres_atributo[prefix]
            with col2:
                for tipo_atributo in data_2[prefix].items():
                    check_key = f"{prefix}_{tipo_atributo[0]}"
                    is_checked = col2.checkbox(tipo_atributo[0], key=check_key, value=bool(data_2[prefix][tipo_atributo[0]]))
                    submited_data[check_key] = int(is_checked)
    with t_dining:
        c_dining = st.container()
        with c_dining:
            prefix = 'dining'
            col1, col2 = st.columns([1,3])

            with col1:
                nombres_atributo[prefix]
            with col2:
                for tipo_atributo in data_2[prefix].items():
                    check_key = f"{prefix}_{tipo_atributo[0]}"
                    is_checked = col2.checkbox(tipo_atributo[0], key=check_key, value=bool(data_2[prefix][tipo_atributo[0]]))
                    submited_data[check_key] = int(is_checked)
    with t_health:
        c_health = st.container()
        with c_health:
            prefix = 'health'
            col1, col2 = st.columns([1,3])

            with col1:
                nombres_atributo[prefix]
            with col2:
                for tipo_atributo in data_2[prefix].items():
                    check_key = f"{prefix}_{tipo_atributo[0]}"
                    is_checked = col2.checkbox(tipo_atributo[0], key=check_key, value=bool(data_2[prefix][tipo_atributo[0]]))
                    submited_data[check_key] = int(is_checked)
    with t_high:
        c_high = st.container()
        with c_high:
            prefix = 'high'
            col1, col2 = st.columns([1,3])

            with col1:
                nombres_atributo[prefix]
            with col2:
                for tipo_atributo in data_2[prefix].items():
                    check_key = f"{prefix}_{tipo_atributo[0]}"
                    is_checked = col2.checkbox(tipo_atributo[0], key=check_key, value=bool(data_2[prefix][tipo_atributo[0]]))
                    submited_data[check_key] = int(is_checked)
    with t_offer:
        c_offer = st.container()
        with c_offer:
            prefix = 'offer'
            col1, col2 = st.columns([1,3])

            with col1:
                nombres_atributo[prefix]
            with col2:
                for tipo_atributo in data_2[prefix].items():
                    check_key = f"{prefix}_{tipo_atributo[0]}"
                    is_checked = col2.checkbox(tipo_atributo[0], key=check_key, value=bool(data_2[prefix][tipo_atributo[0]]))
                    submited_data[check_key] = int(is_checked)

    with t_pay:
        c_pay = st.container()
        with c_pay:
            prefix = 'pay'
            col1, col2 = st.columns([1,3])
            with col1:
                nombres_atributo[prefix]
            with col2:
                for tipo_atributo in data_2[prefix].items():
                    check_key = f"{prefix}_{tipo_atributo[0]}"
                    is_checked = col2.checkbox(tipo_atributo[0], key=check_key, value=bool(data_2[prefix][tipo_atributo[0]]))
                    submited_data[check_key] = int(is_checked)

    with t_popular:
        c_popular = st.container()
        with c_popular:
            prefix = 'popular'
            col1, col2 = st.columns([1,3])
            with col1:
                nombres_atributo[prefix]
            with col2:
                for tipo_atributo in data_2[prefix].items():
                    check_key = f"{prefix}_{tipo_atributo[0]}"
                    is_checked = col2.checkbox(tipo_atributo[0], key=check_key, value=bool(data_2[prefix][tipo_atributo[0]]))
                    submited_data[check_key] = int(is_checked)

    with t_service:
        c_service = st.container()
        with c_service:
            prefix = 'service'
            col1, col2 = st.columns([1,3])
            with col1:
                nombres_atributo[prefix]
            with col2:
                for tipo_atributo in data_2[prefix].items():
                    check_key = f"{prefix}_{tipo_atributo[0]}"
                    is_checked = col2.checkbox(tipo_atributo[0], key=check_key, value=bool(data_2[prefix][tipo_atributo[0]]))
                    submited_data[check_key] = int(is_checked)
    
    submitted = st.form_submit_button("Evaluar")
    features = []

    if submitted:
        df = pd.DataFrame([submited_data])

        cambio = {0: '-', 1: '+'}
        
        for i in range(69):
            if X_subway_proc.iloc[muestra.index, i].values[0] != df.iloc[0, i]:
                # st.write(X_subway_proc.columns[i], X_subway_proc.iloc[muestra.index, i].values[0], df.iloc[0, i])
                features.append((nombres_atributo[X_subway_proc.columns[i].split('_')[0]],X_subway_proc.columns[i].split('_')[1], cambio[df.iloc[0, i]]))

        X_subway_proc.iloc[muestra.index, 0:69] = df.iloc[0, 0:69]

        # actualizado = X_subway_proc.iloc[muestra.index, 1:69].copy()
        actualizado = X_subway_proc.iloc[muestra.index].copy()
        calificacion_modificada = modelo.predict(actualizado)[0]
        st.write(f"Calificación del negocio: {calificacion[calificacion_modificada]}")
        
        st.write(f"Características para modificar")
        if len(features) > 0:
            features_df = pd.DataFrame(data=features)
            features_df.columns=['Tipo','Características', 'Acción']
            st.write(features_df)

        update_calificacion(calificacion[calificacion_modificada])
        if modelo.predict(actualizado)[0] > 0:
            st.balloons()
