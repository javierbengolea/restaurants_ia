import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Ex-stream-ly Cool ML",
    layout="wide")

st.title("Restaurantes y Características")


# Datos proporcionados
@st.cache_data
def read_data():
    X_subway = pd.read_parquet("ML/X_subway.parquet")
    X_subway_proc = pd.read_parquet("ML/X_subway_proc.parquet")
    filename = 'ML/finalized_model.pickle'
    modelo = pickle.load(open(filename, 'rb'))
    return X_subway, X_subway_proc, modelo

X_subway, X_subway_proc, modelo = read_data()

# st.dataframe(X_subway)
# st.dataframe(X_subway.sample(1).T)


# Datos proporcionados
data = {
    'access': [('Wheelchair accessible entrance', 1),
               ('Wheelchair accessible parking lot', 0),
               ('Wheelchair accessible restroom', 0),
               ('Wheelchair accessible seating', 0),
               ('Wheelchair accessible elevator', 0)],
    'amen': [('Bar onsite', 0),
             ('Gender-neutral restroom', 0),
             ('Good for kids', 1),
             ('High chairs', 0),
             ('Restroom', 0),
             ('Wi-Fi', 0)],
    'atmos': [('Cozy', 0), ('Historic', 0), ('Romantic', 0), ('Upscale', 0)],
    'crowd': [('College students', 0),
              ('Family-friendly', 0),
              ('Groups', 0),
              ('Locals', 0),
              ('Tourists', 1),
              ('University students', 0)],
    'dining': [('Breakfast', 0),
               ('Catering', 0),
               ('Dessert', 1),
               ('Dinner', 0),
               ('Lunch', 1),
               ('Seating', 0)],
    'health': [('Mask required', 0),
               ('Reservations required', 0),
               ('Safety dividers at checkout', 0),
               ('Staff get temperature checks', 0),
               ('Staff required to disinfect surfaces between visits', 0),
               ('Staff wear masks', 0),
               ('Temperature check required', 0)],
    'high': [('Fast service', 0),
             ('Great dessert', 0),
             ('LGBTQ friendly', 0),
             ('Live music', 0),
             ('Sports', 0)],
    'offer': [('Alcohol', 0),
              ('Beer', 0),
              ('Cocktails', 0),
              ('Coffee', 1),
              ('Comfort food', 0),
              ('Food', 0),
              ('Halal food', 0),
              ('Happy hour drinks', 0),
              ('Happy hour food', 0),
              ('Healthy options', 1),
              ('Kids menu', 1),
              ('Quick bite', 1),
              ('Vegetarian options', 1),
              ('Wine', 0)],
    'pay': [('Cash-only', 0),
            ('Checks', 0),
            ('Credit cards', 0),
            ('Debit cards', 1),
            ('NFC mobile payments', 0)],
    'popular': [('Breakfast', 0),
                ('Dinner', 1),
                ('Good for working on laptop', 0),
                ('Lunch', 1),
                ('Solo dining', 1)],
    'service': [('Curbside pickup', 0),
                ('Delivery', 1),
                ('Dine-in', 1),
                ('In-store shopping', 0),
                ('Outdoor seating', 0),
                ('Takeout', 1)]
}

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
st.write(f"Calificacion Inicial: {st.session_state['calificacion_restaurant']}")

# Botón para cambiar el valor
# if st.button("Cambiar valor"):
#     update_value()

# # Volver a mostrar el valor actualizado
# st.write(st.session_state['my_value'])

# st.write(f"Calificación Inicial: {calificacion[calificacion_restaurant]}")
# # , muestra['calificacion'].iloc[0])



# a = pd.DataFrame(np.array([['female',1,65.8,18,0,0,'C']]),columns = x_train.columns)



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
                col_2a, col_2b = st.columns(2)
                with col_2a:
                    'Wheelchair accessible entrance'
                    'Wheelchair accessible parking lot'
                    'Wheelchair accessible restroom'
                    'Wheelchair accessible seating'
                    'Wheelchair accessible elevator'
                with col_2b:

                    for tipo_atributo in data_2[prefix].items():
                        check_key = f"{prefix}_{tipo_atributo[0]}"
                        is_checked = col_2b.checkbox("", key=check_key, value=bool(data_2[prefix][tipo_atributo[0]]))
                        submited_data[check_key] = int(is_checked)
                    # st.write(submited_data)

    with t_amen:
        c_amen = st.container()

        with c_amen:
            prefix = 'amen'
            col1, col2 = st.columns([1,3])

            with col1:
                nombres_atributo['amen']
            with col2:
                col_2a, col_2b = st.columns(2)
                with col_2a:
                    'Bar on site'
                    'Gender-neutral restroom'
                    'Good for kids'
                    'High chairs'
                    'Restroom'
                    'Wi-Fi'

                with col_2b:
                    for tipo_atributo in data_2[prefix].items():
                        check_key = f"{prefix}_{tipo_atributo[0]}"
                        is_checked = col_2b.checkbox("", key=check_key, value=bool(data_2[prefix][tipo_atributo[0]]))
                        submited_data[check_key] = int(is_checked)

    with t_atmos:
        c_atmos = st.container()
        with c_atmos:
            prefix = 'atmos'
            col1, col2 = st.columns([1,3])

            with col1:
                nombres_atributo[prefix]
            with col2:
                col_2a, col_2b = st.columns(2)
                with col_2a:
                    for tipo_atributo in data_2[prefix].items():
                        st.write(f'{tipo_atributo[0]}')

                with col_2b:
                    for tipo_atributo in data_2[prefix].items():
                        check_key = f"{prefix}_{tipo_atributo[0]}"
                        is_checked = col_2b.checkbox("", key=check_key, value=bool(data_2[prefix][tipo_atributo[0]]))
                        submited_data[check_key] = int(is_checked)

    with t_crowd:
        c_crowd = st.container()
        with c_crowd:
            prefix = 'crowd'
            col1, col2 = st.columns([1,3])

            with col1:
                nombres_atributo[prefix]
            with col2:
                col_2a, col_2b = st.columns(2)
                with col_2a:
                    for tipo_atributo in data_2[prefix].items():
                        st.write(f'{tipo_atributo[0]}')

                with col_2b:
                    for tipo_atributo in data_2[prefix].items():
                        check_key = f"{prefix}_{tipo_atributo[0]}"
                        is_checked = col_2b.checkbox("", key=check_key, value=bool(data_2[prefix][tipo_atributo[0]]))
                        submited_data[check_key] = int(is_checked)
    with t_dining:
        c_dining = st.container()
        with c_dining:
            prefix = 'dining'
            col1, col2 = st.columns([1,3])

            with col1:
                nombres_atributo[prefix]
            with col2:
                col_2a, col_2b = st.columns(2)
                with col_2a:
                    for tipo_atributo in data_2[prefix].items():
                        st.write(f'{tipo_atributo[0]}')

                with col_2b:
                    for tipo_atributo in data_2[prefix].items():
                        check_key = f"{prefix}_{tipo_atributo[0]}"
                        is_checked = col_2b.checkbox("", key=check_key, value=bool(data_2[prefix][tipo_atributo[0]]))
                        submited_data[check_key] = int(is_checked)
    with t_health:
        c_health = st.container()
        with c_health:
            prefix = 'health'
            col1, col2 = st.columns([1,3])

            with col1:
                nombres_atributo[prefix]
            with col2:
                col_2a, col_2b = st.columns(2)
                with col_2a:
                    for tipo_atributo in data_2[prefix].items():
                        st.write(f'{tipo_atributo[0]}')

                with col_2b:
                    for tipo_atributo in data_2[prefix].items():
                        check_key = f"{prefix}_{tipo_atributo[0]}"
                        is_checked = col_2b.checkbox("", key=check_key, value=bool(data_2[prefix][tipo_atributo[0]]))
                        submited_data[check_key] = int(is_checked)
    with t_high:
        c_high = st.container()
        with c_high:
            prefix = 'high'
            col1, col2 = st.columns([1,3])

            with col1:
                nombres_atributo[prefix]
            with col2:
                col_2a, col_2b = st.columns(2)
                with col_2a:
                    for tipo_atributo in data_2[prefix].items():
                        st.write(f'{tipo_atributo[0]}')

                with col_2b:
                    for tipo_atributo in data_2[prefix].items():
                        check_key = f"{prefix}_{tipo_atributo[0]}"
                        is_checked = col_2b.checkbox("", key=check_key, value=bool(data_2[prefix][tipo_atributo[0]]))
                        submited_data[check_key] = int(is_checked)
    with t_offer:
        c_offer = st.container()
        with c_offer:
            prefix = 'offer'
            col1, col2 = st.columns([1,3])

            with col1:
                nombres_atributo[prefix]
            with col2:
                col_2a, col_2b = st.columns(2)
                with col_2a:
                    for tipo_atributo in data_2[prefix].items():
                        st.write(f'{tipo_atributo[0]}')

                with col_2b:
                    for tipo_atributo in data_2[prefix].items():
                        check_key = f"{prefix}_{tipo_atributo[0]}"
                        is_checked = col_2b.checkbox("", key=check_key, value=bool(data_2[prefix][tipo_atributo[0]]))
                        submited_data[check_key] = int(is_checked)

    with t_pay:
        c_pay = st.container()
        with c_pay:
            prefix = 'pay'
            col1, col2 = st.columns([1,3])

            with col1:
                nombres_atributo[prefix]
            with col2:
                col_2a, col_2b = st.columns(2)
                with col_2a:
                    for tipo_atributo in data_2[prefix].items():
                        st.write(f'{tipo_atributo[0]}')

                with col_2b:
                    for tipo_atributo in data_2[prefix].items():
                        check_key = f"{prefix}_{tipo_atributo[0]}"
                        is_checked = col_2b.checkbox("", key=check_key, value=bool(data_2[prefix][tipo_atributo[0]]))
                        submited_data[check_key] = int(is_checked)
    with t_popular:
        c_popular = st.container()
        with c_popular:
            prefix = 'popular'
            col1, col2 = st.columns([1,3])

            with col1:
                nombres_atributo[prefix]
            with col2:
                col_2a, col_2b = st.columns(2)
                with col_2a:
                    for tipo_atributo in data_2[prefix].items():
                        st.write(f'{tipo_atributo[0]}')

                with col_2b:
                    for tipo_atributo in data_2[prefix].items():
                        check_key = f"{prefix}_{tipo_atributo[0]}"
                        is_checked = col_2b.checkbox("", key=check_key, value=bool(data_2[prefix][tipo_atributo[0]]))
                        submited_data[check_key] = int(is_checked)
    with t_service:
        c_service = st.container()
        with c_service:
            prefix = 'service'
            col1, col2 = st.columns([1,3])

            with col1:
                nombres_atributo[prefix]
            with col2:
                col_2a, col_2b = st.columns(2)
                with col_2a:
                    for tipo_atributo in data_2[prefix].items():
                        st.write(f'{tipo_atributo[0]}')

                with col_2b:
                    for tipo_atributo in data_2[prefix].items():
                        check_key = f"{prefix}_{tipo_atributo[0]}"
                        is_checked = col_2b.checkbox("", key=check_key, value=bool(data_2[prefix][tipo_atributo[0]]))
                        submited_data[check_key] = int(is_checked)
    
    submitted = st.form_submit_button("Evaluar")

    if submitted:
        df = pd.DataFrame([submited_data])

        cambio = {0: '', 1: '+'}
        
        for i in range(69):
            if X_subway_proc.iloc[muestra.index, i].values[0] != df.iloc[0, i]:
                # st.write(X_subway_proc.columns[i], X_subway_proc.iloc[muestra.index, i].values[0], df.iloc[0, i])
                st.write(X_subway_proc.columns[i], cambio[df.iloc[0, i]])

        X_subway_proc.iloc[muestra.index, 0:69] = df.iloc[0, 0:69]

        # actualizado = X_subway_proc.iloc[muestra.index, 1:69].copy()
        actualizado = X_subway_proc.iloc[muestra.index].copy()
        #     st.write("Actualizado:")
        #     st.write(actualizado.T)
        # st.write( calificacion[int(muestra['calificacion'].iloc[0]>= 4.02)])
        # st.write("Luego de las modificaciones: ")
        calificacion_modificada = modelo.predict(actualizado)[0]
        st.write(f"Calificación del negocio: {calificacion[calificacion_modificada]}")
        # calificacion_restaurant = 
        update_calificacion(calificacion[calificacion_modificada])
        if modelo.predict(actualizado)[0] > 0:
            st.balloons()
