import pandas as pd
import streamlit as st
#from pandas import json_normalize
import pickle as pkl
import joblib
import plotly.graph_objects as go
import seaborn as sns
import shap
import lightgbm
from shap.plots import waterfall
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neighbors import NearestNeighbors
    
 
def main():

    
    # -------------------------------------------------------------------------------------------------
   
    #MLFLOW_URI = 'http://192.168.1.86:8501'
    #API_URL ='http://127.0.0.1:5000'
    
    #api_choice = st.sidebar.selectbox('Quelle API souhaitez vous utiliser',['','MLflow'])
    #st.set_page_config(page_icon='ðŸ§Š',layout='centered',initial_sidebar_state='auto')
    
    # Display the title
    st.markdown("""<div style="background-color: Lime; padding:10px; border-radius:10px">
    <h1 style='text-align: center; color: black;'>LOAN APP SCORING DASHBOARD</h1></div>
    <h2 style='text-align: center; color: Lime;'>KHAYREDDINE ROUIBAH.DS</h2>""", unsafe_allow_html=True)
    #---------------------------------------------------------------------------------------------
    # Display the LOGO
    img = Image.open("logo.png")
    st.sidebar.image(img, width=300)
    
    # Display the loan image
    col1, col2, col3 = st.columns(3)
    img = Image.open("original.png")
    #st.image(img, width=200)
    with col1:
        st.image(img, width=700)
    
    # --------------------------------------------------------------------------------------------
    #                          Selected id
    # --------------------------------------------------------------------------------------------
    @st.cache
    def data_load():
        # -----------------------------------------------------------------------------------------------
        #                                   loadings
        # -----------------------------------------------------------------------------------------------
        # Model and threshold loading 
        # ---------------------------
        model     = joblib.load(open("scoring_credit_model.pkl","rb")) 
        threshold = joblib.load('threshold_model.pkl',"rb")
        # -----------------------------------------------------------------------------------------------
        # Data loading
        # -------------
        X_test            = pd.read_csv('X_test_sample.csv')
        X_train           = pd.read_csv('X_train_sample.csv')
        y_test            = pd.read_csv('y_test.csv')
        y_train           = pd.read_csv('y_train.csv')
        expected_values_  = pkl.load(open('expected_value_smpl.pkl', 'rb'))
        shap_values_      = pkl.load(open('shap_vals_smpl.pkl', 'rb'))
        
        return model, threshold, X_test, X_train, y_test, y_train, shap_values_,expected_values_
        
    @st.cache
    def get_id_list():
        # Extract list of all the 'SK_ID_CURR' ids in the X_test dataframe
        customers_id_list = pd.Series(list(X_test.index.sort_values()))  # X_test
        return customers_id_list
    @st.cache
    def get_selected_cust_data(selected_id):
        #selected_id = int(request.args.get('SK_ID_CURR'))
        x_custom   = X_test.iloc[[selected_id]]  # X_test
        y_custom   = y_test.iloc[[selected_id]] # y_test
        return x_custom, y_custom
    @st.cache
    def send_feat_imp():
        feat_imp = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=False)
        return feat_imp
    @st.cache
    def gauge_plot(scor, th):     # Gauge Chart
        scor = int(scor * 100)
        th = int(th * 100)

        if scor >= th:
            couleur_delta = 'red'
        elif scor < th:
            couleur_delta = '#FFFF00'

        if scor >= th:
            valeur_delta = "red"
        elif scor < th:
            valeur_delta = "green"

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=scor,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Selected Customer Score", 'font': {'size': 25}},
            delta={'reference': int(th), 'increasing': {'color': valeur_delta}},
            gauge={
                'axis': {'range': [None, int(100)], 'tickwidth': 1.5, 'tickcolor': "black"},
                'bar': {'color': "#2E00FF"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "black",
                'steps': [
                    {'range': [0, int(th)], 'color': 'Lime'},
                    {'range': [int(th), int(scor)], 'color': couleur_delta}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 1,
                    'value': int(th)}}))

        fig.update_layout(paper_bgcolor="#BFFFFC", font={'color': "darkblue", 'family': "Arial"})
        return fig
    @st.cache
    def get_df_neigh(selected_id_customer):
        # fit nearest neighbors among the selection
        NN = NearestNeighbors(n_neighbors=20)
        NN.fit(X_train)  # X_train_NN
        #X_cust = X_test.loc[[selected_id]]  # X_test
        
        X_cust = X_test.iloc[[selected_id_customer]]  # X_test
        #X_cust = X_cust.drop(["SK_ID_CURR"], axis=1)
        idx = NN.kneighbors(X = X_cust,
                            n_neighbors=20,
                            return_distance=False).ravel()
        nearest_cust_idx = list(X_train.iloc[idx].index)
        # data and target of neighbors
        # ----------------------------
        x_neigh = X_train.loc[nearest_cust_idx, :]
        y_neigh = y_train.loc[nearest_cust_idx]
        return x_neigh, y_neigh   
    @st.cache    
    def shap_value():
        # return the nearest neighbors
        X_neigh, y_neigh = get_df_neigh(selected_id)
        X_cust_= X_test.iloc[[selected_id]]  # X_test
        # prepare the shap values of nearest neighbors + customer
        shap.initjs()
        # creating the TreeExplainer with our model as argument
        explainer = shap.TreeExplainer(model)  # X_train_2.sample(1000)
        # Expected values
        expected_vals = pd.Series(list(explainer.expected_value))
        # calculating the shap values of selected customer
        shap_vals_cust = pd.Series(list(explainer.shap_values(X_cust_)))
        
        # calculating the shap values of neighbors
        shap_val_neigh_ = pd.Series(list(explainer.shap_values(X_neigh)[1]))  # shap_vals[1][X_neigh.index]

        return expected_vals,shap_vals_cust
    def get_list_display_features(f, def_n, key):
        all_feat = f
        n = st.slider("Nb of features to display",
                      min_value=2, max_value=40,
                      value=def_n, step=None, format=None, key=key)

        disp_cols = list(send_feat_imp().sort_values(ascending=False).iloc[:n].index)

        box_cols = st.multiselect('Choose the features to display:',sorted(all_feat), default=disp_cols)
        return box_cols
    
    # find 10000 nearest neighbors among the training set
    @st.cache
    def get_df_thousand_neigh(selected_id_customer):
        # fit nearest neighbors among the selection
        thousand_NN = NearestNeighbors(n_neighbors=500)  # len(X_train)
        thousand_NN.fit(X_train)  # X_train_NN
        X_cust = X_test.iloc[[selected_id_customer]]   # X_test
        idx = thousand_NN.kneighbors(X=X_cust,
                                 n_neighbors=500,  # len(X_train)
                                 return_distance=False).ravel()
        nearest_cust_idx = list(X_train.iloc[idx].index)
        # data and target of neighbors
        # ----------------------------
        x_thousand_neigh = X_train.loc[nearest_cust_idx, :]
        y_thousand_neigh = y_train.loc[nearest_cust_idx]
        return x_thousand_neigh, y_thousand_neigh, X_cust
    # -----------------------   
    # list of customer's ID's
    # -----------------------
    model, threshold, X_test, X_train, y_test, y_train, shap_values_,expected_values_ =   data_load() 
    
    cust_id = get_id_list()
    # -----------------------
    # Selected customer's ID
    # -----------------------
    selected_id = st.sidebar.selectbox('Select customer ID from list:', cust_id, key=1)
    st.write('Your selected ID ...... = ', selected_id)
    # -------------------------------------------------------------------------------
    #                         Customer's data check box
    # -------------------------------------------------------------------------------
 
    if st.sidebar.checkbox("Customer's data"):
        st.markdown('Data of the selected customer :')
        data_selected_cust,y_custom  = get_selected_cust_data(selected_id)
        st.write(data_selected_cust)
    # --------------------------------------------------------------------------------
    #                         Model's decision checkbox
    # --------------------------------------------------------------------------------
    # Compute the score of the customer
    # ---------------------------------
    
    # --------------------------------
    # Probability of selected customer
    # --------------------------------
    
    features = X_train.columns
    x_cust = X_test.iloc[[selected_id]]
    prob = model.predict_proba(x_cust)[:, 1][0]
    
    if st.sidebar.checkbox("Model's decision"):
        # Get score & threshold model
        #score = scoring_cust()
        threshold_model = threshold
        # Display score (default probability)
        st.text('Default probability ............ : {:.0f}%'.format(prob * 100))
        # Display default threshold
        st.text('Default model threshold ........ : {:.0f}%'.format(threshold_model * 100))  #
        # Compute decision according to the best threshold (False= loan accepted, True=loan refused)
        if prob >= threshold_model:
            st.markdown("""<h2 style='text-align: center; color: red;'>Decision : Loan rejected</h2>""", unsafe_allow_html=True)
        else:
            st.markdown("""<h2 style='text-align: center; color: Lime;'>Decision : Loan granted</h2>""", unsafe_allow_html=True)

        #----------------------------------------------------------------------------------
        #              Display customer's gauge meter chart (checkbox)
        #----------------------------------------------------------------------------------
        figure = gauge_plot(prob, threshold_model)
        st.write(figure)
        # Add markdown
        st.markdown("""<h2 style='text-align: center; color: Lime;'>Gauge meter plot for the applicant customer</h2>""", unsafe_allow_html=True)
        
    #if st.checkbox('Classification model infos'):
        expander = st.expander("Classification model infos")
        expander.write("The prediction was made using the Light Gradient Boosting classifier Model\
                        The probability threshold has been personalized and fixed beforehand to penalize the FN (False Negative) in order to avoid \
                        the risk of losing money. If the calculated customer probability is below the threshold, the loan is guaranteed, otherwise \
                        the loan is not guaranteed")
        
        if st.checkbox("Global Feature Importance:"):
            st.markdown("""<h4 style='text-align: center; color: Lime;'>Global Feature Importance</h4>""", unsafe_allow_html=True)
            feat_imp = send_feat_imp().head(20)
            fig, ax = plt.subplots(figsize=(10,8))
            ax = feat_imp.plot(x= 'features',kind = 'bar',color = 'red',figsize=(12,6))
            plt.xticks(rotation=90,fontsize=10)
            plt.yticks(rotation=0,fontsize=10)
            plt.grid(True, color='grey', dashes=(5,2,1,2))
            st.pyplot(fig) 
            st.markdown("""<h4 style='text-align: center; color: Lime;'>SHAP Summary Plot</h4>""", unsafe_allow_html=True)
            
            #if st.sidebar.checkbox('SHAP Summary Plot'):
            fig2, ax2 = plt.subplots(figsize=(10,8))
            ax2 = shap.summary_plot(shap_values_, X_test)
            st.pyplot(fig2)
            
        if st.checkbox('Display local interpretation', key=2):
            with st.spinner('SHAP waterfall plots displaying in progress..... Please wait.......'):
                expected_vals, shap_vals = shap_value()
                nb_features = st.slider("Number of features to display",min_value=2,max_value=50,value=10,step=None,format=None,key=3)
                #st.write(shap_vals)
                #st.write(expected_vals)
                fig2, ax = plt.subplots(figsize=(10,8))
                choice = st.selectbox('Select Shap presentation', ['Selected Id','All Id'])
                if choice =='Selected Id':
                    
                    st.markdown("""<h4 style='text-align: center; color: Lime;'>SHAP Density Scatter Impact On Selected Id </h4>""", unsafe_allow_html=True)
                    ax = shap.summary_plot(shap_vals[1],features=X_test.iloc[[selected_id]], feature_names=features,max_display=nb_features, # nb of displayed features
                                        show=False, color=plt.get_cmap("tab10"))
                    plt.gcf()
                    plt.show()
                    st.pyplot(fig2)
                elif choice=='All Id':
                    st.markdown("""<h4 style='text-align: center; color: Lime;'>SHAP Density Scatter Impact Model Output </h4>""", unsafe_allow_html=True)
                    ax = shap.summary_plot(shap_values_[1],X_test, feature_names=features,max_display=nb_features, # nb of displayed features
                                        show=False, color=plt.get_cmap("tab10"))
                    plt.gcf()
                    plt.show()
                    st.pyplot(fig2)
                # Add markdown
                if st.checkbox("SHAP waterfall Plot"):
                    nb_features_ = st.slider("Number of features to display",min_value=2,max_value=50,value=10,step=None,format=None,key=4)
                    st.markdown("""<h4 style='text-align: center; color: Lime;'>SHAP waterfall Plot for the applicant customer</h4>""", unsafe_allow_html=True)
                    #if st.sidebar.checkbox('SHAP Summary Plot'):
                    fig, ax = plt.subplots(figsize=(10,8))
                    #ax = shap.plots.waterfall(shap_values_[1], )
                    ax = shap.plots._waterfall.waterfall_legacy(expected_vals[0], shap_vals[1][0], max_display=nb_features_, feature_names=features)
                    #ax = shap.waterfall_plot(expected_values_[0], shap_values_[0][selected_id])
                    st.pyplot(fig) 
                    expander = st.expander("Concerning the SHAP waterfall  plot...")
                    expander.write("The above waterfall  plot displays explanations for the individual prediction of the applicant customer.The bottom of a waterfall plot starts as the expected value of the model output \
                    (i.e. the value obtained if no information (features) were provided), and then each row shows how the positive (red) or negative (blue) contribution of \
                    each feature moves the value from the expected model output over the background dataset to the model output for this prediction.")  
                
                if st.checkbox("SHAP Decision Plots"):
                    nb_features__ = st.slider("Number of features to display",min_value=2,max_value=50,value=10,step=None,format=None,key=5)
                    st.markdown("""<h4 style='text-align: center; color: Lime;'>SHAP Decision Plots</h4>""", unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(10,8))
                    ax = shap.decision_plot(expected_vals[0],shap_vals[1][0],feature_names=list(features),feature_display_range=slice(None, -1-nb_features__ , -1))
                    st.pyplot(fig)
                    # Add details title
                    expander = st.expander("Concerning SHAP Decision Plots")
                    expander.write('SHAP decision plots show how complex models arrive at their predictions (i.e., how models make decisions). \
                    This graph illustrates decision plot features and use cases with simple examples. ')
    # ---------------------------------------------------------------------------------
    #                 Display local SHAP waterfall checkbox
    # ---------------------------------------------------------------------------------
    # find 20 nearest neighbors among the training set
    # ------------------------------------------------
 
        if st.checkbox('Features Distribution By Class', key=6):
                    #st.header('Boxplots of the main features')
            
            fig, ax = plt.subplots(figsize=(20, 10))
            with st.spinner('Boxplot creation in progress...please wait.....'):
                    # Get Shap values for customer
                expected_vals, shap_vals = shap_value()
                    #shap_vals, expected_vals = values_shap(selected_id)
                    # Get features names
                    #features = feat()
                    # Get selected columns
                disp_box_cols = get_list_display_features(features, 2, key=7)
                    # -----------------------------------------------------------------------------------------------
                    # Get tagets and data for : all customers + Applicant customer + 20 neighbors of selected customer
                    # -----------------------------------------------------------------------------------------------
                
                
                    # neighbors + Applicant customer :
                data_neigh, target_neigh = get_df_neigh(selected_id)
                
                data_thousand_neigh, target_thousand_neigh, x_customer = get_df_thousand_neigh(selected_id)

                x_cust, y_cust = get_selected_cust_data(selected_id)
                x_customer.columns = x_customer.columns.str.split('.').str[0]
                    # Target impuatation (0 : 'repaid (....), 1 : not repaid (....)
                    # -------------------------------------------------------------
                target_neigh = target_neigh.replace({0: 'repaid (neighbors)',
                                                     1: 'not repaid (neighbors)'})
                target_thousand_neigh = target_thousand_neigh.replace({0: 'repaid (neighbors)',
                                                                       1: 'not repaid (neighbors)'})
                y_cust = y_cust.replace({0: 'repaid (customer)',
                                         1: 'not repaid (customer)'})

                    # y_cust.rename(columns={'10006':'TARGET'}, inplace=True)
                    # ------------------------------
                    # Get 1000 neighbors personal data
                    # ------------------------------
                df_thousand_neigh = pd.concat([data_thousand_neigh[disp_box_cols], target_thousand_neigh], axis=1)
                df_melt_thousand_neigh = df_thousand_neigh.reset_index()
                df_melt_thousand_neigh.columns = ['index'] + list(df_melt_thousand_neigh.columns)[1:]
                df_melt_thousand_neigh = df_melt_thousand_neigh.melt(id_vars=['index', 'TARGET'],
                                                                     value_vars=disp_box_cols,
                                                                     var_name="variables",  # "variables",
                                                                     value_name="values")
                                                                     
                st.markdown("""<h4 style='text-align: center; color: Lime;'>Boxplots Of The Main Features </h4>""", unsafe_allow_html=True)
                sns.set_palette("bright")
                sns.boxplot(data=df_melt_thousand_neigh, x='variables', y='values',
                hue='TARGET', linewidth=1, width=0.4, showfliers=False,
                saturation=0.5, ax=ax)
                
                    # ------------------------------
                    # Get 20 neighbors personal data
                    # ------------------------------
                df_neigh = pd.concat([data_neigh[disp_box_cols], target_neigh], axis=1)
                df_melt_neigh = df_neigh.reset_index()
                df_melt_neigh.columns = ['index'] + list(df_melt_neigh.columns)[1:]
                df_melt_neigh = df_melt_neigh.melt(id_vars=['index', 'TARGET'],
                                                   value_vars=disp_box_cols,
                                                   var_name="variables",  # "variables",
                                                   value_name="values")
                st.write(df_melt_neigh)
                sns.swarmplot(data=df_melt_neigh, x='variables', y='values', hue='TARGET', linewidth=1,
                              palette=['lightgreen', 'darkred'], marker='o', size=15, edgecolor='k', ax=ax)
                    # -----------------------
                    # Applicant customer data
                    # -----------------------
                df_selected_cust = pd.concat([x_customer[disp_box_cols], y_cust], axis=1)
                    # st.write("df_sel_cust :", df_sel_cust)
                df_melt_sel_cust = df_selected_cust.reset_index()
                df_melt_sel_cust.columns = ['index'] + list(df_melt_sel_cust.columns)[1:]
                df_melt_sel_cust = df_melt_sel_cust.melt(id_vars=['index', 'TARGET'],
                                                         value_vars=disp_box_cols,
                                                         var_name="variables",
                                                         value_name="values")

                sns.swarmplot(data=df_melt_sel_cust, x='variables', y='values',
                              linewidth=1, color='y', marker='o', size=20,
                              edgecolor='k', label='applicant customer', ax=ax)

                    # legend
                h, _ = ax.get_legend_handles_labels()
                ax.legend(handles=h[:5])

                plt.xticks(rotation=20, ha='right')
                plt.show()

                st.pyplot(fig)  

                plt.xticks(rotation=20, ha='right')
                plt.show()
                st.markdown("""<h6 style='text-align: center; color: Lime;'>Distribution Of Features By Class And the 20 nearest neighbors of applicant customer</h6>""", unsafe_allow_html=True)
                

                expander = st.expander("Dispersion Graph")
                expander.write("These boxplots show the distribution of the preprocessed features values\
                            used by the model to make a prediction. \
                            The green boxplot are for the customers that repaid their loan, and red boxplots are for the customers that didn't repay it.Over the boxplots are\
                            superimposed (markers) the values\
                            of the features for the 20 nearest neighbors of the applicant customer in the training set. The \
                            color of the markers indicate whether or not these neighbors repaid their loan. \
                            Values for the applicant customer are superimposed in yellow.")
            
    # ---------------------------------------------------------------------------------------
   
if __name__ == '__main__':
    main()
