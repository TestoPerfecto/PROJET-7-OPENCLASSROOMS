# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:56:25 2023

@author: PERFECTO
"""

# -*- coding: utf-8 -*-

import streamlit as st
#import streamlit-shap as st_shap
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

import joblib
import shap
from xplotter.insights import plot_aggregation

shap.initjs()
shap.getjs()
# from streamlit.hashing import _CodeHasher

st.set_option('deprecation.showPyplotGlobalUse', False)
#df = pd.read_csv("Housing.csv")

PATH = 'C:/Users/PERFECTO/'

# Chargeons les datasets
X_train_1 = pd.read_csv(PATH+'X_train_smtomek_sd.csv' )
y_train_1 = pd.read_csv(PATH+'y_train_smtomek_sd.csv' )
X_valid_1 = pd.read_csv(PATH+'X_valid_smtomek_sd.csv' )
y_valid_1 = pd.read_csv(PATH+'y_valid_smtomek_sd.csv' )

X_test = pd.read_csv(PATH+"X_test_feat.csv")


# Chargeons le modèle

mon_best_model = joblib.load('model_7.pkl')

## La prédiction
y_pred_model = mon_best_model.predict(X_test)
y_pred_model_df = pd.DataFrame(y_pred_model, columns=['y_pred_test'])

# y_pred_model_proba est la probabilité que l'instance de données appartienne à chaque classe.
y_pred_model_proba = mon_best_model.predict_proba(X_test)
#y_pred_model_proba

## y_pred_model_proba en DataFrame
y_pred_model_proba_df = pd.DataFrame(y_pred_model_proba, columns=['proba_classe_0', 'proba_classe_1'])

## Il n'y a que 391 personnes qui représent un défaut de paiement de plus de 90%
#y_pred_model_proba_df[y_pred_model_proba_df['proba_classe_1'] > 0.9].sort_values(by='proba_classe_1', ascending=False)

# Les valeurs SHAP estiment l'impact d'une fonctionnalité sur les prédictions 
# tandis que l'importance des fonctionnalités estime l'impact d'une 
# fonctionnalité sur l'ajustement du modèle.

shap.initjs()
explainer = shap.TreeExplainer(mon_best_model)
shap_values = explainer.shap_values(X_test)

# Shap_values pour la classe 0 et la classe 1
#shap_values

# Shap_values uniquement pour la classe 1
#shap_values[1]

# shap_values_df correspond au DataFrame issu de shap_values[1]
shap_values_df = pd.DataFrame(data=shap_values[1], columns=X_test.columns)
#shap_values_df.shape



data_groupes = pd.concat([y_pred_model_proba_df['proba_classe_1'], shap_values_df], axis=1)
#data_groupes.shape

# On décide de former 5 groupes
pd.qcut(data_groupes.proba_classe_1, precision=1, q=5).head(10)

# On labellise les 5 groupes
data_groupes['classe_clients'] = pd.qcut(data_groupes.proba_classe_1,q=5,
                                   precision=1, labels=['7%_et_moins',
                                                  '8%_10%',
                                                  '11%_20%',
                                                  '21%_50%',
                                                  '51%_et_plus'])
# On pratique la moyenne de chaque classe
data_groupes_mean = data_groupes.groupby(['classe_clients']).mean()
data_groupes_mean = data_groupes_mean.rename_axis('classe_clients').reset_index()                                                  
le_max = X_test.shape[0]-1                                                 
                                                  




st.sidebar.title("Sommaire")
pages = ["Demande de crédit","Identification du client","Analyse globale","Analyse finale", "Décision finale"]

page = st.sidebar.radio("Aller vers la page : ", pages)
if page == pages[0]:
    st.write("### Demande de crédit")
    st.write("##### La société financière 'Prêt à dépenser' propose des crédits à la consommation ")
    st.write("##### Vous avez sollicité un prêt auprès de notre société en remplissant un formulaire de renseignement.") 
    st.write("##### Nous allons ensemble étudier votre dossier et vous apporter une réponse.") 
    st.write("##### Toute réponse vous sera expliquée avec la plus grande clarté.")
    st.image("Credit_score.png")
    
elif page == pages[1]:
    st.write("### Identification du client") 
    
    idx = st.number_input("Rappel de votre identifiant", min_value=-1, max_value=le_max , placeholder="Tapez ici votre identifiant...")
        
    if idx == -1:
        st.write('##### Merci de nous indiquer votre identifiant')             
    elif idx >= 0:
         st.write('#### Votre identifiant est :', idx)
         st.write("#### Nous procédons à l'analyse de votre dossier numéroté", idx)
    st.image("remise_identifiant1.png")
    
elif page == pages[2]:    
    st.write("### Analyses globales")   
    st.write("##### Notre analyse algorithmique scinde nos clients en 5 groupes")
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_aggregation(df=data_groupes,
                 group_col='classe_clients',
                 value_col='proba_classe_1',
                 aggreg='mean',
                 ax=ax, orient='v', size_labels=12)
    sns.color_palette("husl", 9)
    plt.title("Probabilité Moyenne de Défaut de Paiement par Groupe de Clients\n",
          loc="center", fontsize=18, fontstyle='italic')
    st.pyplot(fig)
    st.write()
    idx = st.number_input("Rappel de votre identifiant", min_value=0, max_value=le_max , placeholder="Tapez ici votre identifiant...")
    
    if st.checkbox("##### Afficher le groupe du client"):        
        shap_du_client = data_groupes[data_groupes.index == idx]
        groupe_du_client = shap_du_client['classe_clients'].values
        st.write("##### Nos analyses révèlent que vous faites partie du groupe : ",groupe_du_client[0] )
        st.write()
        st.write()
        if groupe_du_client[0] == "7%_et_moins":
            def_pay = round(data_groupes_mean.iloc[0]["proba_classe_1"]*100,2)
            st.write("##### Votre groupe représente un défaut de paiement en moyenne de {}%".format(def_pay))
        if groupe_du_client[0] == "8%_10%":           
            def_pay = round(data_groupes_mean.iloc[1]["proba_classe_1"]*100,2)
            st.write("##### Votre groupe représente un défaut de paiement en moyenne de {}%".format(def_pay))
        if groupe_du_client[0] == "11%_20%":
            def_pay = round(data_groupes_mean.iloc[2]["proba_classe_1"]*100,2)
            st.write("##### Votre groupe représente un défaut de paiement en moyenne de {}%".format(def_pay))
        if groupe_du_client[0] == "21%_50%":
            def_pay = round(data_groupes_mean.iloc[3]["proba_classe_1"]*100,2)
            st.write("##### Votre groupe représente un défaut de paiement en moyenne de {}%".format(def_pay))
        if groupe_du_client[0] == "51%_et_plus":
            def_pay = round(data_groupes_mean.iloc[4]["proba_classe_1"]*100,2)
            st.write("##### Votre groupe représente un défaut de paiement en moyenne de {}%".format(def_pay))
    
    if st.checkbox("##### Afficher les analyses concernant le groupe du client"): 
        st.write("##### Ce graphique s'intitule 'beeswarm'.")
        st.write("##### Les variables sont classées en fonction de leur effet sur la prédiction")
        fig, ax = plt.subplots(figsize=(6, 4))
        the_group = data_groupes[data_groupes['classe_clients']==groupe_du_client[0]].drop(labels=['classe_clients', "proba_classe_1"], axis=1)
        shap_values_group = explainer.shap_values(the_group)
        shap.summary_plot(shap_values_group[1], the_group)
        st.pyplot(fig)
    
elif page == pages[3]:
    st.write("### Analyse finale")
    st.write("##### Cette partie présente les éléments qui ont influé sur la décision finale.")
    st.write("##### La première analyse révèle les éléments constitutifs des caractéristiques du client")
    st.write("##### La seconde analyse compare les résultats du client à la moyenne de chacun des autres.")
    st.write()
    idx = st.number_input("Rappel de votre identifiant", min_value=0, max_value=le_max , placeholder="Tapez ici votre identifiant...")
    if st.checkbox("#### Analyse sur le client"):
        st.write("##### Les deux figures montrent comment les caractéristiques ont influencé la prédiction ")
        st.write()
        st.write("##### La première visualisation explique comment le modèle est arrivé à la prédiction faite ")
        #fig, ax = plt.subplots(figsize=(6, 4))
        st_shap(shap.force_plot(explainer.expected_value[1], 
                shap_values[1][idx,:], 
                X_test[X_test.index == idx], 
                link='logit',
                figsize=(20, 8),
                ordering_keys=True,
                text_rotation=0,
                contribution_threshold=0.05))
        
        #st.pyplot(fig)
        st.write("##### La deuxième visualisation aide à identifier les principales caractéristiques ayant un pouvoir de décision élevé dans le modèle au niveau individuel")
        st.write("##### Ce graphique est une meilleure visualisation de l'importance des caractéristiques de tous les prédicteurs chez chaque individu.")        
        fig, ax = plt.subplots(figsize=(6, 4))
        shap.decision_plot(explainer.expected_value[1], 
                   shap_values[1][idx,:], 
                   X_test[X_test.index == idx], 
                   feature_names=X_test.columns.to_list(),
                   feature_order='importance',
                   #feature_display_range=slice(None, -15, -1),
                   link='logit')
        st.pyplot(fig)
    if st.checkbox("#### Analyse comparative des groupes avec le client"):  
         st.write("##### Chaque groupe est représenté par sa moyenne")
         st.write()
         st.write("##### Meilleure représentation graphique pour les groupes et le client")
         comparaison_client_groupe = pd.concat([data_groupes[data_groupes.index == idx], 
                                      data_groupes_mean],axis = 0)
                                      
         comparaison_client_groupe['classe_clients'] = np.where(comparaison_client_groupe.index == idx, 
                                                          X_test.iloc[idx, 0],comparaison_client_groupe['classe_clients'])
         comparaison_client_groupe_light = comparaison_client_groupe.drop(labels=['classe_clients', "proba_classe_1"], axis=1)
         comparaison_client_groupe_numpy = comparaison_client_groupe.drop(labels=['classe_clients', "proba_classe_1"], axis=1).to_numpy()
         les_groupes_plus = ['Client','Groupe 1 avec 7%_et_moins', 'Groupe 2 avec 8%_10%', 'Groupe 3 avec 11%_20%', 'Groupe 4 avec 21%_50%',
                             'Groupe 5 avec 51%_et_plus']
         fig, ax = plt.subplots(figsize=(6, 4))
         shap.decision_plot(explainer.expected_value[1], 
                   comparaison_client_groupe_numpy, 
                   feature_names=comparaison_client_groupe.drop(labels=['classe_clients', "proba_classe_1"], axis=1).columns.to_list(),
                   feature_order='importance',
                   highlight=0,
                   legend_labels=les_groupes_plus,
                   legend_location='lower left',
                   feature_display_range=slice(None, -57, -1),
                   link='logit')
         st.pyplot(fig)
    
                                                              
elif page == pages[4]:
    st.write("## Décision finale")
    idx = st.number_input("Rappel de votre identifiant", min_value=0, max_value=le_max , placeholder="Tapez ici votre identifiant...")
    st.write('### Votre identifiant est : ', idx)
    result = round(y_pred_model_proba[idx][1]*100,2)
    st.write("##### L'analyse algorithmique révèle un défaut de paiement de {}%".format(result))
    if result >=20.0:
        st.write("### Votre demande de crédit est REFUSÉE")
    else:
        st.write("### Votre demande de crédit est ACCEPTÉE")
    
   
    
   
    
   
    
   
   
   