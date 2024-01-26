
##########################################################
################ I Import des données ####################
##########################################################
import pandas as pd

df = pd.read_csv('dataset.csv')
df.head()
df.info()

##########################################################
################ II Nettoyage des données ################
##########################################################

# Sampling
df = df.sample(n = 10000)


def cleaning(df):
    # Drop Duplicates
    df.drop_duplicates(inplace=True)

    # Row Selection

    # Column Selection
    df = df.drop(columns=[])

    # Deal with NA
    df = df.dropna(subset=['MIS_Status']) 
    
    # Data formating for training data (outside pipe)

    return df

df = cleaning(df)

##########################################################
################ III Train / Test / Split  ###############
##########################################################


from sklearn.model_selection import train_test_split
X = df.drop('charges', axis=1)
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False,test_size=0.2, random_state=42)


##########################################################
############### IV Outliers################ ##############
##########################################################

from sklearn.ensemble import IsolationForest #, LocalOutlierFactor
anomalie = IsolationForest(random_state=0, contamination=0.02)
anomalie.fit(X_train)
not_outliers = anomalie.predict(X_train) == 1 
X_train = X_train[not_outliers]

##########################################################
############### IV Construction du Pipeline ##############
##########################################################

##### Les transformers
# .fit() pour les entrainer
# .transform pour appliquer la transformation
# .fit_transform pour faire les deux

## Encodage
## Normalisation
## Imputation
## Selection
## Extraction



############### IV.a Numeric features ##############

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures, Binarizer, KBinsDiscretizer
from sklearn.impute import SimpleImputer


numeric_features = ["age","bmi", "children"]

## PCA

from sklearn.preprocessing import  StandardScaler
std = StandardScaler()
X_train_standard = std.fit_transform(X_train[numeric_features])

from sklearn.decomposition import PCA
# Choisir le nb de composants
n_dims = X_train_standard.shape[1]
model = PCA(n_components=n_dims)
model.fit(X_train_standard)

variances = model.explained_variance_ratio_

meilleur_dims = np.argmax(np.cumsum(variances) > 0.90)

import matplotlib.pyplot as plt
plt.bar(range(n_dims), np.cumsum(variances))
plt.hlines(0.90, 0, meilleur_dims, colors='r')
plt.vlines(meilleur_dims, 0, 0.90, colors='r')

## reste du pipe

numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
         ('poly', PolynomialFeatures(2))   Avant la normalisation car sinon pas même echelle.

        ('stdscaler', StandardScaler()),  # moyenne nulle et écart type = 1 -> Reg, SVM, PCA
        # ('robust', RobustScaler())  # moins sensible aux outliers (soustrait par médian et divise par inter-quartile)
        # ('minmax', MinMaxScaler())  # entre 0 et 1 -> dessente gradient
        
        # ('pca', PCA(n_components=2)) # après standardisation

        # ('bin', Binarizer(threshold= 8))  # coupe en deux
        # ('kbin', KBinsDiscretizer(n_bins= 4))  # coupe en plus
        
        ])




############### IV.b Ordonal features ##############
ordinal_features = [ "Exter Qual",  "Kitchen Qual"]

from sklearn.preprocessing import OrdinalEncoder
exter_cat = [ 'Po', 'Fa','TA', 'Gd','Ex']
kitchen_cat = [ 'Po', 'Fa','TA', 'Gd',"Ex"]

from sklearn.preprocessing import OneHotEncoder
ordinal_transformer = OrdinalEncoder(categories=[exter_cat, kitchen_cat])


############### IV.c Categorial features ##############
categorial_features = [ "sex", "region", "smoker"]

from sklearn.preprocessing import OneHotEncoder
categorical_transformer = OneHotEncoder(sparse=True)


############## Specific Columns  ##########
from sklearn.compose import make_column_selector
import numpy as np
numeric = make_column_selector(dtype_include=np.number)

############### IV.d Combinaison ##############
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('ord', ordinal_transformer, ordinal_features),
        ('cat', categorical_transformer, categorial_features)
    ],
    remainder="passthrough" 
)

"""
Si pour un type de variable, on a pas de transformer, il faut utiliser
remainder="passthrough"      
"""

############### IV.d FeatureSelector ##############

from sklearn.feature_selection import VarianceThreshold
feature_selector = VarianceThreshold(threshold=0.2)

from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression
feature_selector = SelectKBest(f_classif, k=10)
feature_selector.scores_ # score du test d'independance
feature_selector.pvalues_

feature_selector.get_support() # quelles variables ont été selectionnées


from sklearn.feature_selection import RFECV
from sklearn.linear_model import SGDClassifier
feature_selector = RFECV(SGDClassifier(random_state=0), step=1, min_features_to_select=2, cv=5)
feature_selector.grid_scores_ # les scores successifs
feature_selector.ranking_ # classement des variables

from sklearn.feature_selection import SelectFromModel
feature_selector = SelectFromModel(SGDClassifier(random_state=0), threshold='mean')





############### IV.d Estimator ##############
from sklearn.linear_model import LinearRegression
reg = LinearRegression()


############### IV.d Final_pipe ##############

from sklearn.pipeline import Pipeline
pipe = Pipeline([
     ('preprocessor', preprocessor),
     ('feature_selector', feature_selector),

     ('clf', reg)
])


##########################################################
################ V Train and evaluate ####################
##########################################################


############### V.a Y processing ##############

## LabelEncoder pour les ordinals, Labelbinarizer pour les non ordinals (à moitier vrai)
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
     
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)
     
y_label_back = encoder.inverse_transform(y_test)

############### V.a Train and Evaluate Function ##############

def run_experiment(model,modele_type,confusion=False,mlflow_tracking=False):
    
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    try:
        print('Best Hyperparameters: %s' % model.best_params_)
    except:
        pass

    if modele_type == "regression":
        from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
        print("######## R^2 : ")
        print("TRAIN :",r2_score(y_train, y_pred_train))
        print("TEST :",r2_score(y_test, y_pred_test))
        if mlflow_tracking:
            mlflow.log_metric("train_r2", r2_score(y_train, y_pred_train))
            mlflow.log_metric("test_r2", r2_score(y_test, y_pred_test))
        print("######## MAE : ")
        print("TRAIN :",mean_absolute_error(y_train, y_pred_train))
        print("TEST :",mean_absolute_error(y_test, y_pred_test))
        if mlflow_tracking:
            mlflow.log_metric("train_mae", mean_absolute_error(y_train, y_pred_train))
            mlflow.log_metric("test_mae", mean_absolute_error(y_test, y_pred_test))
        print("######## MSE : ")
        print("TRAIN :",mean_squared_error(y_train, y_pred_train))
        print("TEST :",mean_squared_error(y_test, y_pred_test))
        if mlflow_tracking:
            mlflow.log_metric("train_mse", mean_squared_error(y_train, y_pred_train))
            mlflow.log_metric("test_mse", mean_squared_error(y_test, y_pred_test))
        
        

    elif modele_type == "classification":
        from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        print("######## accuracy_score : ")
        print("TRAIN :",accuracy_score(y_train, y_pred_train))
        print("TEST :",accuracy_score(y_test, y_pred_test))
        if mlflow_tracking:
            mlflow.log_metric("train_accuracy", accuracy_score(y_train, y_pred_train))
            mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred_test))
        print("######## f1_score : ")
        print("TRAIN :",f1_score(y_train, y_pred_train))
        print("TEST :",f1_score(y_test, y_pred_test))
        if mlflow_tracking:
            mlflow.log_metric("train_f1", f1_score(y_train, y_pred_train))
            mlflow.log_metric("test_f1", f1_score(y_test, y_pred_test))
        print("######## precision_score : ")
        print("TRAIN :",precision_score(y_train, y_pred_train))
        print("TEST :",precision_score(y_test, y_pred_test))
        if mlflow_tracking:
            mlflow.log_metric("train_precision", precision_score(y_train, y_pred_train))
            mlflow.log_metric("test_precision", precision_score(y_test, y_pred_test))
        print("######## recall_score : ")    
        print("TRAIN :",recall_score(y_train, y_pred_train))
        print("TEST :",recall_score(y_test, y_pred_test))
        if mlflow_tracking:
            mlflow.log_metric("train_recall", recall_score(y_train, y_pred_train))
            mlflow.log_metric("test_recall", recall_score(y_test, y_pred_test))
        print("######## roc_auc_score : ")    
        print("TRAIN :",roc_auc_score(y_train, y_pred_train))
        print("TEST :",roc_auc_score(y_test, y_pred_test))  
        if mlflow_tracking:
            mlflow.log_metric("train_roc", roc_auc_score(y_train, y_pred_train))
            mlflow.log_metric("test_roc", roc_auc_score(y_test, y_pred_test))  

        if confusion:
            
            if mlflow_tracking:
                mlflow.log_artifact("test_confusion", confusion_matrix(y_test, y_pred_test)) 
            return model.best_estimator_, confusion_matrix(y_test, y_pred_test)

    return model.best_estimator_

############### V.b Cross Validation ##############

from sklearn.model_selection import KFold

# define evaluation
cv = KFold(n_splits=10, random_state=1)

############### V.c Hyparameters space setting ##############

import numpy as np

# define search space
space = dict()
space['clf__list_hyperparameter'] = ["squared_error", "absolute_error"]
space['clf__discrete_hyperparameter'] = np.arange(30,150,10)  
space['clf__continuous_hyperparameter'] = np.linspace(0,1000,100) 



############### V.d RandomSearch and GridSearc ##############

# ou scoring = "accuracy"
from sklearn.model_selection import RandomizedSearchCV
model = RandomizedSearchCV(pipe, space, n_iter=1000, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv, random_state=1,verbose=2)

from sklearn.model_selection import GridSearchCV
model = GridSearchCV(pipe, space, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv)

import mlflow
try:
    experiment_id = mlflow.get_experiment_by_name("experiment_name").experiment_id
except AttributeError:
    experiment_id = mlflow.create_experiment("experiment_name")


with mlflow.start_run(experiment_id=experiment_id) as run:
    for key,value in space.items()
        mlflow.log_param("lr", 0.01)
    
    model_fit = run_experiment(model, "regression")

############### V.e ValidationCurve ##############

from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt

k = np.arange(1, 50)

train_score, val_score = validation_curve(pipe, X_train, y_train,
                                          'clf__discrete_hyperparameter', k, cv=5)

plt.plot(k, val_score.mean(axis=1), label='validation')
plt.plot(k, train_score.mean(axis=1), label='train')

plt.ylabel('score')
plt.xlabel('clf__discrete_hyperparameter')
plt.legend()

############### V.f LearningCurve ##############

# best_model_params = model_fit["clf"].get_params()
# xg_best = XGBClassifier(**best_model_params)
# best_pipe = Pipeline([
#      ('preprocessor', preprocessor),
#      ('clf', xg_best)
# ])

from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

N, train_score, val_score = learning_curve(best_pipe, X_train, y_train,cv=cv)


print(N)
plt.plot(N, train_score.mean(axis=1), label='train')
plt.plot(N, pd.DataFrame(val_score).mean(axis=1), label='validation')
plt.xlabel('train_sizes')
plt.legend()

##########################################################
################ V Feature Importance ####################
##########################################################


model_estimator = model_fit.named_steps["clf"]

column_names= model_pipe[0].get_feature_names_out()
feature_importance = pd.Series(model_estimator.feature_importances_, index=column_names)
plot_importances_df =\
        feature_importance\
        .nlargest(10)\
        .sort_values()\
        .to_frame('value')\
        .rename_axis('feature')\
        .reset_index()

import plotly.express as px
fig = px.bar(plot_importances_df, 
                x='value', 
                y='feature')
fig.update_layout(title_text="feature importance", title_x=0.5) 
fig.update(layout_showlegend=False)

fig.show()


##########################################################
################ V MLFlow ####################
##########################################################

import mlflow
try:
    experiment_id = mlflow.get_experiment_by_name("loan_analysis").experiment_id
except AttributeError:
    experiment_id = mlflow.create_experiment("loan_analysis")

import mlflow
from mlflow.models.signature import infer_signature
run_name = "grid_no_tuning_to_deploy"


with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
    # Log the baseline model to MLflow
    model_fit = grid_pipe.fit(X_train, y_train)
    
    for param,value in model_fit.best_estimator_[-1].get_params().items():
        mlflow.log_param(param, value)
    
    mlflow.log_param("PCA", True) 

    # Log model
    
    signature = infer_signature(X_train, model_fit.best_estimator_.predict(X_train))

    
    mlflow.sklearn.log_model(model_fit.best_estimator_, "xgboost_with_PCA", signature=signature)

    model_uri = mlflow.get_artifact_uri("xgboost_with_PCA")
    
    
    eval_data = X_test
    eval_data["label"] = y_test

    # Evaluate the logged model
    result = mlflow.evaluate(
        model_uri,
        eval_data,
        targets="label",
        model_type="classifier",
        evaluators=["default"],
    )