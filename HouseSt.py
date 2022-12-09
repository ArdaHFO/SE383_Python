import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from requests.api import options
import streamlit as st
import xgboost as xgb
from sklearn.base import TransformerMixin
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
import joblib
from sklearn.pipeline import Pipeline
from PIL import Image

st.set_page_config(page_title='SE383 Python Project', page_icon=':house_with_garden')

#Hide Menü
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)


#Working as Single Page
padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

detail = False
col1, col2, col3 = st.columns([1,1,1])

class SparseMatrix(TransformerMixin):
    def __init__(self):
        None
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        categorical_columns = ['MSSubClass','MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'Electrical', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish','GarageQual','GarageCond', 'PavedDrive', 'Fence', 'SaleType', 'SaleCondition']
        X[categorical_columns] = X[categorical_columns].astype(str)
        ohe = joblib.load('ohe.joblib')
        hot = ohe.transform(X[categorical_columns].astype(str))
        cold_df = X.select_dtypes(exclude=["object"])
        cold = csr_matrix(cold_df.values)
        final_sparse_matrix = hstack((hot, cold))
        final_csr_matrix = final_sparse_matrix.tocsr()
        return final_csr_matrix

data_pipeline = Pipeline([('sparse', SparseMatrix())])

bst = xgb.Booster()

bst.load_model('housepricexgb_final.model')

single_row = pd.read_csv('single_row.csv',index_col =0)


if "load_state" not in st.session_state:
     st.session_state.load_state = False

if "load_state" not in st.session_state:
    st.session_state.load_state = False
if (st.sidebar.button("Let's start!") or st.session_state.load_state) != True : 
    col1, col2, col3 = st.columns([1,1,1])
    st.title("W E L C O M E !")

    st.title("Please select the properties of the house you want!")
    st.title("About")

    st.markdown('<span style="font-family:Papyrus; font-size:1.5em;">It is a Python project for SE383 Python Programming class that shows the approximate price of a house with features.</span>', unsafe_allow_html=True)

    st.markdown('<span style="font-family:Papyrus; font-size:1.5em;">This Project allows consumers to make a prediction about the price in terms of the features they want. At the same time, it is helpful to see giving up which features will get a more suitable price.</span>', unsafe_allow_html=True)

    st.markdown('<span style="font-family:Papyrus; font-size:1.5em;">Select the features from the left-hand pop-up window to see the price of the house</span>', unsafe_allow_html=True)

    st.markdown(" ")
    st.title("Project Developers")
    st.markdown(" ")
    #our sudent numbers and Names
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.markdown("<b><i>Gamze Çelik</i></b><br>180704023", unsafe_allow_html=True)
    with col2:
        st.markdown("<b><i>M.Yavuz Gökmen</i></b><br>190704004", unsafe_allow_html=True)
    with col3:
        st.markdown("<b><i>Arda Hacıfevzioğlu</i></b><br>200704041", unsafe_allow_html=True)

else:
    st.session_state.load_state = True
    st.title("Please select the properties of the house you want!")
    with st.sidebar:
            #CAT_OPTIONS             

            ###
        LandContour_options = ['','Near Flat/Level', 'Banked - Quick and significant rise','Hillside','Depression']
        LandContour_dict = {'':'','Near Flat/Level':'Lvl', 'Banked - Quick and significant rise':'Bnk','Hillside':'HLS','Depression':'Low'}
        LandContour = st.sidebar.selectbox('Flatness of the property',options=LandContour_options)

        PavedDrive_options=['', 'Paved','Partial Pavement','Dirt/Gravel']
        PavedDrive_dict={'':'', 'Paved':'Y','Partial Pavement':'P','Dirt/Gravel':'N'}
        PavedDrive= st.sidebar.selectbox('Paved driveway',options=PavedDrive_options)          

        BsmtQual_options = ['','Excellent', 'Good', 'Typical', 'Fair']  
        BsmtQual_dict = {'':'','Excellent':'5','Good':'4', 'Typical':'3', 'Fair':'2'}
        BsmtQual = st.sidebar.selectbox('Type of Basement Quality',options=BsmtQual_options)

        Condition1_options = ['','Adjacent to arterial street','Adjacent to feeder street', 'Normal', "Within 200' of North-South Railroad", 'Adjacent to North-South Railroad','Near positive off-site feature--park, greenbelt, etc.','Adjacent to postive off-site feature',"Within 200' of East-West Railroad",'Adjacent to East-West Railroad']
        Condition1_dict = {'':'','Adjacent to arterial street':'Artery','Adjacent to feeder street':'Feedr', 'Normal':'Norm', "Within 200' of North-South Railroad":'RRNn', 'Adjacent to North-South Railroad':'RRAn','Near positive off-site feature--park, greenbelt, etc.':'PosN','Adjacent to postive off-site feature':'PosA',"Within 200' of East-West Railroad":'RRNe','Adjacent to East-West Railroad':'RRAe'}
        Condition1 = st.sidebar.selectbox('Proximity to various conditions',options=Condition1_options)
                       
        FireplaceQu_options = ['','Exceptional Masonry Fireplace','Masonry Fireplace in main level','Prefabricated Fireplace in main living area or Masonry Fireplace in basement','Prefabricated Fireplace in basement','Ben Franklin Stove'] #TA ? 
        FireplaceQu_dict = {'':'','Exceptional Masonry Fireplace':'5', 'Masonry Fireplace in main level':'4','Prefabricated Fireplace in main living area or Masonry Fireplace in basement':'3','Prefabricated Fireplace in basement':'2','Ben Franklin Stove':'1'}
        FireplaceQu = st.sidebar.selectbox('Fireplace Quality',options=FireplaceQu_options)
             
        Alley_options = ['','Gravel','Paved']
        Alley_dict = {'':"",'Gravel':'Grvl','Paved':'Pave'}
        Alley = st.sidebar.selectbox('Type of alley access to property', options=Alley_options)

        Neighborhood_options = ['',"Bloomington Heights","Bluestem",'Briardale','Brookside','Clear Creek','College Creek','Crawford','Edwards','Gilbert','Iowa DOT and Rail Road', 'Meadow Village', 'Mitchell', 'North Ames', 'Northridge', 'Northpark Villa', 'Northridge Heights', 'Northwest Ames', 'Old Town', 'South & West of Iowa State University', 'Sawyer', 'Sawyer West', 'Somerset', 'Stone Brook', 'Timberland', 'Veenker']

        Neighborhood_dict = {'':'',
            'Bloomington Heights':'Blmngtn',
            'Bluestem':'Blueste',
            'Briardale':'BrDale',
            'Brookside':'BrkSide',
            'Clear Creek':'ClearCr',
            'College Creek':'CollgCr',
            'Crawford':'Crawfor',
            'Edwards':'Edwards',
            'Gilbert':'Gilbert',
            'Iowa DOT and Rail Road':'IDOTRR',
            'Meadow Village':'MeadowV', 
            'Mitchell':'Mitchel', 
            'North Ames':'Names', 
            'Northridge':'NoRidge', 
            'Northpark Villa':'NPkVill', 
            'Northridge Heights':'NridgHt', 
            'Northwest Ames':'NWAmes', 
            'Old Town':'OldTown', 
            'South & West of Iowa State University':'SWISU', 
            'Sawyer':'Sawyer', 
            'Sawyer West':'SawyerW', 
            'Somerset':'Somerst', 
            'Stone Brook':'StoneBr', 
            'Timberland':'Timber', 
            'Veenker':'Veenker'}

        Neighborhood = st.sidebar.selectbox('Physical locations within Ames city limits',options=Neighborhood_options)

        GarageQual_options = ['','Excellent', 'Good', 'Typical', 'Fair', 'Poor']  
        GarageQual_dict = {'':'','Excellent':'5','Good':'4', 'Typical':'3', 'Fair':'2','Poor':'1'}
        GarageQual = st.sidebar.selectbox('Garage Quality',options=GarageQual_options)

        # categoric ones
        
        ###


        #NUM_OPTIONS
            
                        
        BsmtFinSF1 = st.sidebar.select_slider(
                'Finished Basement Area',
                options=[*range(0, 5645)])

        FullBath = st.sidebar.select_slider(
                'Full bathrooms above grade',
                options=[*range(0,4)])
            
        KitchenQual= st.sidebar.select_slider(
                'Kitchen quality',
                options=[*range(2,6)])
            
        TotalBsmtSF = st.sidebar.select_slider(
                'Total square feet of basement area',
                options=[*range(0,6111)])
            
        ndFlrSF = st.sidebar.select_slider(
                'Second floor square feet',
                options=[*range(1,2066)])
            
        GarageArea = st.sidebar.select_slider(
                'Size of garage in square feet',
                options=[*range(0,1419)])
            
        KitchenAbvGr = st.sidebar.select_slider(
                'Kitchens above grade' ,
                options = [*range(0,4)])
            
        OverallQual = st.sidebar.select_slider(
                'Rates the overall material and finish of the house',
                options=[*range(1,11)])

        GrLivArea = st.sidebar.select_slider(
                'Above grade (ground) living area square feet',
                options=[*range(334,5643)])
                
        GarageCars = st.sidebar.select_slider(
                'The size of garage according to car capacity',
                options=[*range(0,5)])   

        ExterQual = st.sidebar.select_slider(
                'Evaluates the quality of the material on the exterior',
                options=[*range(2,6)])

        
        # numeric ones

        soru_list1 = [LandContour_dict,PavedDrive_dict,BsmtQual_dict,Condition1_dict,FireplaceQu_dict,Alley_dict,Neighborhood_dict,GarageQual_dict] #Categoric olanların dict'leri bu listeye eklenecek
        soru_list2 = [LandContour,PavedDrive,BsmtQual,Condition1,FireplaceQu,Alley,Neighborhood,GarageQual] #categoric olanların seçimleri bu listeye eklenecek
        soru_list3 = []
        for i,j in zip(soru_list2,soru_list1):
            soru_list3.append(j[i])
        soru_list3 += [BsmtFinSF1,FullBath,KitchenQual,TotalBsmtSF,ndFlrSF,GarageArea,KitchenAbvGr,OverallQual,GrLivArea,GarageCars,ExterQual] #numeric olanların seçimleri bu listeye eklenecek

        soru_list4 = ['LandContour','PavedDrive','BsmtQual','Condition1','FireplaceQu','Alley','Neighborhood','GarageQual','BsmtFinSF1','FullBath','KitchenQual','TotalBsmtSF','2ndFlrSF','GarageArea','KitchenAbvGr','OverallQual','GrLivArea','GarageCars','ExterQual'] #categoric ve numeric olanların isimleri bu listeye eklenecek

    if st.sidebar.button('Show House Price'):
        for i in range(len(soru_list3)):
            if soru_list3[i] != '' :
                single_row.loc[0,soru_list4[i]] = soru_list3[i]
        single_row_transformed = data_pipeline.fit_transform(single_row)
        xgmat = xgb.DMatrix(single_row_transformed,missing = -999.0)
        ypred = bst.predict(xgmat)
        st.title('With selected properties estimated Price of the House: ')
        st.title(np.round(ypred[0]))
        if st.button('Start over') :
            st.legacy_caching.clear_cache()
            del st.session_state.load_state
            st.experimental_rerun()






detail = st.sidebar.checkbox('Details')

if detail:
        # Create a page dropdown 
        st.write(" ")
        st.write(" ")     
        st.title('Model Evaluation and Success Metrics')
        st.write('General Information about Model:')

        col1,col3,col5  = st.columns([1,3,3])        

        im1 = Image.open('MSE-İterations.png').resize((500,300))    
        
    
        im2 = Image.open('original_predicted.png').resize((450,305))
        
        
        im3 = Image.open('Prices.png')
                
       

        with col1:
                b1 = st.image(im1, width=400)              
        
        with col5:
                b3 = st.image(im3, width=400)

        with col1:               
                b2 = st.image(im2, width=400)      
        
        with col5:
                st.write("MSE: 635030379.79")
                st.write(" RMSE: 25199.81")
                st.write(" R2_score: 0.904")