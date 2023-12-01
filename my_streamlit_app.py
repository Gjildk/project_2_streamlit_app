import streamlit as st 
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from streamlit_option_menu import option_menu
import plotly.express as px 
from IPython.display import YouTubeVideo

# tranfert du dataset pour le top10
url = "FINAL_BD.csv"
df_top10 = pd.read_csv(url)


#transfert des datasets graphique poour le dashboard imdb 



#ML


features =['genres', 'liste_titres_FR', 'stars', 'director_name']

def combine_features(row) :
  return row['liste_titres_FR']+' '+row['genres']+' '+row['director_name']+' '+row['stars']

def titre_lower():
    df_top10['liste_titres_FR'] = df_top10['liste_titres_FR'].lower()
    
for feature in features:
    df_top10[feature] = df_top10[feature].fillna('')
    
df_top10['combined_features'] = df_top10.apply(combine_features, axis = 1)
df_top10['index'] = df_top10.index

def get_title_from_index(index):
  return df_top10[df_top10.index == index]['liste_titres_FR'].values[0]
def get_index_from_title(Title):
  return df_top10[df_top10.Title == Title]['tconst'].values[0]

#fonction top10 
def top_movies_by_year(df_top10, year):
    movies_for_year = df_top10[df_top10['startYear'] == year]
    top_movies = movies_for_year.sort_values(by='averageRating', ascending=False).head(10)
    return top_movies[['liste_titres_FR', 'averageRating', 'image_url']]

# """
# STREAMLIT APP 
# """



# """
# TITRE
# ""
# Création du titre centré
st.markdown("<h1 style='text-align: center;'>ALLO SAVVY</h1>", unsafe_allow_html=True)


# ONGLET 0 = ACCUEIL
def onglet0():
    # st.markdown("<h1 style='text-align: center;'>Projet2</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Présentation</h3>", unsafe_allow_html=True)
    st.write("<h3 style='text-align: center;'>- Jeanne - Nassim- Louka</h3>", unsafe_allow_html=True)
    st.write('Un cinéma en perte de vitesse situé dans la Creuse vous contacte. Il a décidé de passer le cap du digital en créant un site Internet taillé pour les locaux. Pour aller encore plus loin, il vous demande de créer un moteur de recommandations de films qui à erme, enverra des notifications aux clients via Internet.')
    st.write("                                                                                             ")
    st.markdown("<h3 style='text-align: center;'>Mission</h3>", unsafe_allow_html=True)
    st.write('Crée un  moteur de recommandations de films pour votre site internet. Cette application sera mise à disposition des clients de votre cinéma afin de leur proposer un service supplémentaire, en ligne, en plus du cinéma classique.')
    st.markdown("<h3 style='text-align: center;'>Informations sur la Creuse</h3>", unsafe_allow_html=True)
    st.image("Creuse.PNG")
    population_par_age = {
    'Tranche d\'âge': ['0 à 14 ans', '15 à 29 ans', '30 à 44 ans', '45 à 59 ans', '60 à 74 ans', '75 ans ou +'],
    '2020': [13.2, 12.3, 14.4, 21.1, 24.0, 14.9]
}

    df1 = pd.DataFrame(population_par_age)

    fig = px.bar(df1, x='Tranche d\'âge', y='2020', labels={'2020': 'Population (%)'}, title='Population en Creuse par tranche d\'âge en 2020 (%)')
    fig.update_traces(marker_color='orange')

    fig.update_layout(
        annotations=[
            dict(
                text="Sources : Insee, RP2009, RP2014 et RP2020, exploitations principales, géographie au 01/01/2023. (%)",
                xref="paper", yref="paper",
                x=0, y=-0.25,
                showarrow=False,
                align="left",
                valign="middle"
            )
        ]
    )
    st.plotly_chart(fig)


    data_internet_2012 = {
        'Tranche d\'âge': ['15-29 ans', '30-44 ans', '45-59 ans', '60-74 ans', '75 ans et plus'],
        'Internet': [97.7, 92.2, 82.3, 52.2, 16.5],
        'Internet mobile': [75.0, 50.8, 33.7, 16.4, 3.1]
    }
    df2 = pd.DataFrame(data_internet_2012)

    st.write('Fréquence d\'utilisationd d\'Internet en France au cours des 3 derniers mois (%)')
    st.write(df2)
    st.markdown("*Source : Insee, enquête Technologies de l'information et de la communication 2012.*")
    
    
    
    
    
# ONNGLET 1 = DASHBOARD donnée IMDB
def onglet1():
    # st.markdown("<h1 style='text-align: center;'>Projet2</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Dashboard</h3>", unsafe_allow_html=True)
    st.write('Dataset - title.basics.tsv.gz')
    percent_basics = pd.read_csv(r'percent_basics.csv', sep = ',',low_memory=False)
    fig1 = px.bar(percent_basics, x='TitleType', y='Percentage',
                labels={'TitleType': 'Type de vidéo', 'Percentage': 'Pourcentage'},
                title='Types de vidéo présents dans le dataset (%)')

    fig1.update_traces(marker_color='orange')

    fig1.update_layout(xaxis=dict(tickangle=45))
    fig1.update_yaxes(title='Pourcentage') 
    st.plotly_chart(fig1)

    st.write('Suppression des films +18')
    plus_18_count= pd.read_csv('plus_18_count.csv', sep = ',')
    plus_18_count['isAdult'] = plus_18_count['isAdult'].replace({'yes': 'oui', 'no': 'non'})
    fig1_2 = px.bar(plus_18_count, x='isAdult', y='number', labels={'isAdult': 'Films +18', 'number': 'Nombre'},
                title='Nombre de films interdits aux moins de 18 ans', color_discrete_sequence=['orange'])
    st.plotly_chart(fig1_2)


    st.write('Sélection du type movie')
    movies_per_year = pd.read_csv(r'movies_per_year.csv', sep = ',',low_memory=False)
    movies_per_year['Year'] = pd.to_datetime(movies_per_year['Year'], format='%Y')
    years = pd.date_range(start=movies_per_year['Year'].min(), end=movies_per_year['Year'].max(), freq='10Y')

    fig2 = px.line(movies_per_year, x='Year', y='Number_of_Films', 
                labels={'Year': 'Année de sortie', 'Number_of_Films': 'Nombre de films'},
                title='Evolution du nombre de films au cours du temps')

    fig2.update_traces(mode='markers+lines', marker=dict(symbol='circle', size=8, color='orange'),
                    line=dict(color='orange', width=2))

    fig2.update_layout(xaxis=dict(title='Année de sortie', tickvals=years, tickformat="%Y"),
                    yaxis=dict(title='Nombre de films'))

    st.plotly_chart(fig2)

    st.write('Sélection d\'une durée maximale')
    films_runtime_df = pd.read_csv('films_runtime_df.csv', sep = ',')
    labels = films_runtime_df['Catégorie']
    sizes = films_runtime_df['Pourcentage']
    colors = ['black', 'orange']

    fig3 = px.pie(names=labels, values=sizes, title='Répartition des films au-dessus et en-dessous de 200 minutes')
    fig3.update_traces(marker=dict(colors=colors))
    st.plotly_chart(fig3)

    st.write('Sélection d\'une durée minimale')
    rt= pd.read_csv('rt_distribution_df.csv', sep = ',')

    mean_rt = rt["Runtime"].mean()
    mode_rt = rt["Runtime"].value_counts().idxmax()

    fig4 = px.box(rt, y='Runtime', title="Boxplot de la durée des films (Durée < 200 minutes) avec outliers",
                width=500, height=700)

    fig4.add_annotation(x=-0.35, y=mean_rt -2 , text=f"Mean: {mean_rt:.2f}", showarrow=False, font=dict(color='black'))
    fig4.add_annotation(x=-0.35, y=mode_rt +3, text=f"Mode: {mode_rt}", showarrow=False, font=dict(color='black'))
    fig4.update_traces(marker=dict(color='orange'))
    fig4.update_layout(yaxis=dict(dtick=25))
    fig4.update_layout(yaxis=dict(title="Durée (minutes)"))
    st.plotly_chart(fig4)

    st.write('Dataset - title.ratings.tsv.gz')
    ratings= pd.read_csv('ratings.csv', sep = ',')
    fig5 = px.histogram(ratings, x='averageRating', nbins=30, labels={'averageRating': 'Notes', 'count': 'Fréquence'},
                    title='Distribution des notes moyennes')
    fig5.update_traces(marker=dict(color='orange'))  # Changer la couleur des barres en orange
    fig5.update_xaxes(tickvals=list(range(11)))
    st.plotly_chart(fig5)

    votes = pd.read_csv('votes.csv', sep = ',')
    votes = votes.drop(columns='Unnamed: 0')
    st.write(votes.describe())

    st.write('Dataset - title.akas.tsv.gz')
    country_counts_df= pd.read_csv('country_counts_df.csv', sep = ',')
    top_countries = country_counts_df.head(10)
    fig6 = px.bar(top_countries, x='Film_Count', y='Country', orientation='h', 
                labels={'Film_Count': 'Nombre total de films diffusés', 'Country': 'Pays'},
                title='Top 10 des pays qui diffusent le plus de films',
                text='Film_Count') 

    fig6.update_traces(marker_color='orange', textposition='outside', textfont_size=12)
    fig6.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig6)

    st.write('Dataset - title.principals.tsv.gz')
    category_counts_df = pd.read_csv('category_counts_df.csv', sep = ',')
    category_counts_df = category_counts_df[category_counts_df['Category'] != 'category']
    fig7 = px.bar(category_counts_df, x='Count', y='Category', orientation='h',
                title='Barres horizontales des catégories')
    fig7.update_layout(title='Répartition des différents membres du casting',
                    xaxis_title='Nombre de personnes',
                    yaxis_title='Profession')
    fig7.update_layout(yaxis=dict(autorange="reversed"))
    fig7.update_traces(marker=dict(color='orange'))
    st.plotly_chart(fig7)
    
# ONNGLET 2 = Données Recommandations
def onglet2():
    final = df_top10.copy()
    st.markdown("<h3 style='text-align: center;'>Dashboard</h3>", unsafe_allow_html=True)
    #final = 'le dataset de LOUKA'

    ##genre_final_au cours du temps
    # Récupération de toutes les années uniques dans l'ordre
    final['startYear'] = final['startYear'].round().astype(int)
    final['genres_split'] = final['genres'].str.split(',')
    final2 = final.explode('genres_split')
    genre_counts = final2['genres_split'].value_counts()
    genre_counts2 = genre_counts.head(7)

    all_years = final['startYear'].sort_values().unique()

    # Récupération de tous les genres uniques
    all_genres = genre_counts2.index

    # Ajout d'une option "Tous les films" à la liste des genres
    all_genres_list = ['Tous les genres'] + all_genres.tolist()

    # Ajout de widgets pour la sélection du genre
    selected_genre = st.selectbox('Sélectionner un genre', all_genres_list)

    # Filtrer les données en fonction du genre sélectionné
    if selected_genre != 'Tous les genres':
        filtered_data = final[final['genres_split'].apply(lambda x: selected_genre in x)]
    else:
        filtered_data = final.copy()

    # Création d'un DataFrame pour stocker les données de chaque genre
    genre_year_count = pd.DataFrame(columns=['startYear', 'count', 'Genre'])

    # Génération des données pour chaque genre
    for genre in all_genres:
        if genre != 'Tous les genres':
            genre_data = filtered_data[filtered_data['genres_split'].apply(lambda x: genre in x)]
            films_per_year = genre_data.groupby('startYear').size().reset_index(name='count')
            films_per_year['Genre'] = genre
            genre_year_count = pd.concat([genre_year_count, films_per_year])

    # Création du graphique avec Plotly Express pour tous les genres
    if selected_genre == 'Tous les genres':
        fig = px.line(genre_year_count, x='startYear', y='count', color='Genre',
                    labels={'startYear': 'Année', 'count': 'Nombre de films', 'Genre': 'Genre'},
                    title='Evolution du nombre de films au cours du temps')
        fig.update_layout(legend_title='Top Genres')
        
    else:
        # Création du graphique avec Plotly Express pour un seul genre sélectionné
        films_per_year = filtered_data.groupby('startYear').size().reset_index(name='count')
        fig = px.line(films_per_year, x='startYear', y='count',
                    labels={'startYear': 'Année', 'count': 'Nombre de films'},
                    title=f'Evolution du nombre de films au cours du temps pour le genre {selected_genre}')
        fig.update_traces(line=dict(color='orange'))
        
    st.plotly_chart(fig)



    ##ratings_final
    mean_rt = final["runtimeMinutes"].mean()
    mode_rt = final["runtimeMinutes"].value_counts().idxmax()

    fig2 = px.box(final, y='runtimeMinutes', 
                title="Boxplot de la durée des films (Durée < 201 minutes) avec outliers",
                labels={'runtimeMinutes': 'Durée (minutes)'},width=500, height=700)
    fig2.add_annotation(x=-0.35, y=mean_rt + 2, text=f"Mean: {mean_rt:.2f}", showarrow=False, font=dict(color='black'))
    fig2.add_annotation(x=-0.35, y=mode_rt + 0.15, text=f"Mode: {mode_rt}", showarrow=False, font=dict(color='black'))
    fig2.update_traces(boxmean=True, marker=dict(color='orange'))
    fig2.update_layout(yaxis=dict(dtick=25))

    st.plotly_chart(fig2)



    ##top profession_finale
    # Convertir vers des listes les colonnes 'actors, actress et directors. Nettoyer les écritures :

    def convert_to_list(column_value, separator=','):
        if isinstance(column_value, int):
            return []
        # Supprimer les crochets et les guillemets
        column_value = column_value.replace('[', '').replace(']', '').replace("'", "")
        # Diviser la chaîne en une liste en utilisant le séparateur spécifié
        values_list = column_value.split(separator)
        # Retirer les espaces en début et fin de chaque valeur
        values_list = [value.strip() for value in values_list]
        return values_list

    final['actor_name'] = final['actor_name'].apply(lambda x: convert_to_list(x, ','))
    final['actress_name'] = final['actress_name'].apply(lambda x: convert_to_list(x, ','))
    final['director_name'] = final['director_name'].apply(lambda x: convert_to_list(x, ','))
    final['spoken_languages'] = final['spoken_languages'].apply(lambda x: convert_to_list(x, ','))
    final['stars'] = final['stars'].apply(lambda x: convert_to_list(x, ','))

    #top acteurs_final
    final3 = final.explode('actor_name')
    actor_counts = final3['actor_name'].value_counts()
    top_actors = actor_counts.head(11)

    top_actors_sans_unknown = dict(list(top_actors.items())[1:])

    #top actrice_final
    final4 = final.explode('actress_name')
    actress_counts = final4['actress_name'].value_counts()
    top_actress = actress_counts.head(11)
    top_actress_sans_unknown = dict(list(top_actress.items())[1:])

    #top directors_final
    final5 = final.explode('director_name')
    director_counts = final5['director_name'].value_counts()
    top_directors = director_counts.head(10)

    options = ['Acteurs', 'Actrices', 'Réalisateurs']
    selected_option = st.selectbox('Sélectionner une catégorie', options)

    if selected_option == 'Acteurs':
        top_actors_sans_unknown = dict(list(top_actors.items())[1:])
        fig3 = px.bar(
            x=list(top_actors_sans_unknown.keys()),
            y=list(top_actors_sans_unknown.values()),
            labels={'x': 'Acteur', 'y': 'Nombre de films'},
            title="Acteurs les plus représentés"
        )
        
    elif selected_option == 'Actrices':
        top_actress_sans_unknown = dict(list(top_actress.items())[1:])
        fig3 = px.bar(
            x=list(top_actress_sans_unknown.keys()),
            y=list(top_actress_sans_unknown.values()),
            labels={'x': 'Actrices', 'y': 'Nombre de films'},
            title="Actrices les plus représentées"
        )
    else:
        fig3 = px.bar(
            x=top_directors.index,
            y=top_directors.values,
            labels={'x': 'Réalisateurs', 'y': 'Nombre de films'},
            title="Réalisateurs les plus représentés"
        )
    fig3.update_traces(marker=dict(color='orange'))
    fig3.update_layout(yaxis=dict(range=[0, 50])) 
    
    st.plotly_chart(fig3)


    ##top film par genre_final

    # Sélectionner les 15 meilleurs genres
    top_genres = genre_counts.head(15).index
    df_top_genres = final2[final2['genres_split'].isin(top_genres)]

    # Trier le DataFrame par note moyenne en ordre décroissant
    df_top_genres = df_top_genres.sort_values(by='popularity', ascending=False)

    selected_genre = st.selectbox('Sélectionner un genre', top_genres)

    # Filtrer le DataFrame pour le genre sélectionné
    top_3_movies_genre = df_top_genres[df_top_genres['genres_split'] == selected_genre].head(3)

    # Afficher le top 3 des films du genre sélectionné
    if not top_3_movies_genre.empty:
        fig4 = px.bar(
            top_3_movies_genre,
            x='popularity',
            y='liste_titres_FR',
            orientation='h', 
            labels={'popularity': 'Note de popularité','liste_titres_FR': '' },
            title=f"Top 3 des films pour le genre {selected_genre}"
        )
        fig4.update_xaxes(range=[7, 10], gridcolor='grey', showgrid=True)
        fig4.update_yaxes(categoryorder='total ascending')
        fig4.update_traces(marker=dict(color='orange'))
        fig4.update_layout(
            title=dict(font=dict(color='black')),
            plot_bgcolor='white',
            paper_bgcolor='white')
        st.plotly_chart(fig4)
    else:
        st.write(f"Aucun film trouvé pour le genre {selected_genre}.")





# ONGLET 3 = SYSTEME DE RECOMMANDATION 
def onglet3():
    # st.markdown("<h1 style='text-align: center;'>Projet2</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Systeme de recommendation</h3>", unsafe_allow_html=True)
    df_top10['averageRating'] = df_top10['averageRating'].astype("float")
    
    #enlever les unknown 
    df_top10['stars'] = df_top10['stars'].replace('unknown', '', regex=True)

   # Fonction pour trouver l'index du film dans le DataFrame
    def find_movie_index(df_top10, movie_user_likes):
        movie_index = df_top10[df_top10['liste_titres_FR'] == movie_user_likes].index
        return movie_index if not movie_index.empty else []
    
    

    tfidf = TfidfVectorizer()
    count_matrix = tfidf.fit_transform(df_top10['liste_titres_FR'] + ' ' +df_top10['director_name'] +' '+ df_top10['stars'] +' '+ df_top10['genres'] + ' '+df_top10['overview_clean'])
    
    # Réduire la dimension de la matrice 
    svd = TruncatedSVD(n_components=100)  
    tfidf_matrix = svd.fit_transform(count_matrix)

    # Initialisation du modèle Nearest Neighbors, pour éviter les erreurs de dimensions (avec les voisins)
    nn_model = NearestNeighbors(n_neighbors=4, metric='cosine')
    nn_model.fit(tfidf_matrix)

    # Utilisateur spécifie le film, le réalisateur et/ou d'autres critères | Ajoutez un widget de sélection pour choisir l'année
    movie_user_likes = st.selectbox("Entrez le titre du film que vous aimez : ",(df_top10['liste_titres_FR'].sort_values().unique()), index=None)
    
    
    # # Trouver l'index du film de l'utilisateur dans le DataFrame
    movie_index = find_movie_index(df_top10, movie_user_likes)
    
    if not movie_user_likes:
        st.write("Veuillez entrer un titre de film.")  
        
    else:
    
        if any(movie_index):
            movie_index = movie_index[0]  
                        
            # Trouver les voisins les plus proches avec Nearest Neighbors
            _,similiar_movies_indices = nn_model.kneighbors([tfidf_matrix[movie_index]])

            st.write("Films recommandés :")
            for index in similiar_movies_indices[0][1:]:  # Ignorer le film lui-même
                    col1, col2 = st.columns([1, 3])  # Ratio 1:3 pour la largeur de la colonne

                    # Colonne de gauche (col1) pour l'image
                    col1.image(df_top10.loc[index, 'image_url'], width=420)

                    # Colonne de droite (col2) pour les informations
                    with col2:
                        st.write('<p class="col2-content"><strong>Recommendation :</strong> ', df_top10.loc[index, 'liste_titres_FR'], '</p>', unsafe_allow_html=True)
                        st.write('<p class="col2-content"><strong>Note du film :</strong> {}</p>'.format(df_top10.loc[index, 'averageRating']), unsafe_allow_html=True)
                        st.write('<p class="col2-content"><strong>Réalisateur :</strong> ', df_top10.loc[index, 'director_name'], '</p>', unsafe_allow_html=True)
                        st.write('<p class="col2-content"><strong>Casting :</strong> ', df_top10.loc[index, 'stars'], '</p>', unsafe_allow_html=True)
                        st.write('<p class="col2-content"><strong>Genres :</strong> ', df_top10.loc[index, 'genres'], '</p>', unsafe_allow_html=True)
                        
                        # Vidéo YouTube intégrée avec un bouton de lecture
                        trailer_link = df_top10.loc[index, 'trailer_link_youtube']
                        video_id = trailer_link.split("v=")[1]
                        st.video(f"https://www.youtube.com/watch?v={video_id}")
                        
                        st.write('<p class="col2-content"><strong>Description :</strong> ', df_top10.loc[index, 'overview'], '</p>', unsafe_allow_html=True)
                        st.write('<p class="col2-content">------------------------</p>', unsafe_allow_html=True)
                    
                # ajouter du css pour décaler la col2
                    st.markdown(f""" <style>div[data-testid="stHorizontalBlock"] {{margin-left: -100px;}}</style>""", unsafe_allow_html=True)
                
                # Ajouter du CSS pour décaler l'image
                    st.markdown(f"""<style>div[data-testid="stImage"] {{margin-left: -250px;}}</style>""", unsafe_allow_html=True)
                    st.markdown(f"""<style>div[data-testid="stImage"] {{margin-top: 24px;}}</style>""", unsafe_allow_html=True)
        else:
            st.write("Film non trouvé dans la base de données.")



# ONGLET 4 = TOP 10 DES FILMS PAR ANNEES
def onglet4():
    # st.markdown("<h1 style='text-align: center;'>Projet2</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Top 10 des films par années</h3>", unsafe_allow_html=True)  
    
    # Convertissez la colonne startYear en type numérique
    df_top10['startYear'] = df_top10['startYear'].round().astype(int)
    
    # Ajoutez un widget de sélection pour choisir l'année
    selected_year = st.selectbox("Entrer une année", (df_top10['startYear'].sort_values().unique()), index=None)

    # Appelez la fonction top_movies_by_year avec le DataFrame et l'année sélectionnée
    result = top_movies_by_year(df_top10, selected_year)
    
    # Utilisez le titre sélectionné en dehors de la boucle pour éviter d'afficher plusieurs boutons
    selected_movie = None

    # Vérifiez si le bouton a déjà été cliqué pour éviter l'itération infinie
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False

    # Affichez les images avec la colonne "image_url"
    for index, row in result.iterrows():
        # Utilisez un lien pour rendre l'image cliquable
        if not st.session_state.button_clicked and st.button(f"Voir détails pour {row['liste_titres_FR']}"):
            selected_movie = row['liste_titres_FR']
            st.session_state.button_clicked = True  # Marquez le bouton comme cliqué

        st.image(row['image_url'], caption=f"{row['liste_titres_FR']} - Note moyenne : {row['averageRating']}", use_column_width=True)
        
    st.markdown(f"""<style>div[data-testid="stImage"] {{margin-left: 0px;}}</style>""", unsafe_allow_html=True)

  
# LA NAVIGATION
def main():
    with st.sidebar:
                # Charger l'image
        st.image("banniere.png")

        # Appliquer le style CSS avec une balise Markdown
        st.markdown('<style>img {width: 5000px; height: auto;}</style>', unsafe_allow_html=True)
        st.markdown('<style>img {width: 5000px;}</style>', unsafe_allow_html=True)
        # Appliquer le style CSS avec une balise Markdown pour déplacer l'image vers le haut
        st.markdown("<style>img {margin-top: -50px;}</style>",unsafe_allow_html=True)
        

        onglet_selectionne = option_menu("NAVIGATION", ["Présentation", "Données IMDb", "Données Recommandations", "Systeme de recommendation", "Top 10 des films par années"], 
            icons=['house', 'gear', 'gear'], menu_icon="cast", default_index=0)
   

    if onglet_selectionne == "Présentation":
        onglet0()
    elif onglet_selectionne == "Données IMDb":
        with st.spinner("Chargement..."):
            time.sleep(2)
        onglet1()
    elif onglet_selectionne == "Données Recommandations":
        with st.spinner("Chargement... En route vers le top 10 des films par l'année de votre choix YOUHOU!"):
            time.sleep(2)
        onglet2()
    elif onglet_selectionne == "Systeme de recommendation":
        with st.spinner("Chargement... Vers votre sélection de film..."):
            time.sleep(2)
        onglet3()    
    elif onglet_selectionne == "Top 10 des films par années":
        with st.spinner("Chargement... En route vers le top 10 des films par l'année de votre choix YOUHOU!"):
            time.sleep(2)
        onglet4() 
    else:
        print("erreur")

if __name__ == "__main__":
    main()