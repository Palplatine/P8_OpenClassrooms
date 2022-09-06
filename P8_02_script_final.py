# Nos imports
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.ml.linalg import VectorUDT
from pyspark.sql import Row
from pyspark.sql.functions import split
from pyspark.sql.functions import col
from pyspark.sql.functions import udf
from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import PandasUDFType
from pyspark.sql.types import IntegerType, StringType, ArrayType

import boto3

import pandas as pd
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

###########################################################################################
###########################################################################################
###########################################################################################

if __name__ == '__main__':

    # On crée notre environnement Spark
    spark = SparkSession.builder.appName('P8_OCR_VLE').getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    ###########################################################################################
    ###########################################################################################
    ###########################################################################################

    # Nos fonctions :
    def load_img(img_path):
        """
        Prend le chemin dans un bucket S3 d'un objet (une image), le lit et
        le transforme sous la forme d'une liste.

                Parameters:
                    img_path : le chemin de l'image dans notre bucket S3
        """
        
        # On récupère nos images en se connectant à notre bucket S3
        s3 = boto3.resource('s3', region_name='eu-west-3')
        bucket = s3.Bucket('p8ocrvle')
        
        # On récupère les objets dans notre bucket
        img = img_path.replace('s3://p8ocrvle/', '')
        bucket_file = bucket.Object(img)
        s3_response = bucket_file.get()
        file_stream = s3_response['Body']
        image = Image.open(file_stream)
        
        # On enlève les images qui ne retournent rien
        if image is None:
            image = 0
            
        else:
            image = np.asarray(image)
            image = np.resize(image, (299, 299, 3))
            image = image.flatten().tolist()
        
        return image

    def model_pretrained():
        """
        Retourne notre modèle (InceptionV3), sans la couche la plus haute
        (couche de classification) et on y ajoute manuellement les poids pré-entraînés.
        """
        
        model = InceptionV3(weights=None, include_top=False)
        model.set_weights(model_weights.value)
        return model

    def model_pretrained():
        """
        Retourne notre modèle (InceptionV3), sans la couche la plus haute
        (couche de classification) et on y ajoute manuellement les poids pré-entraînés.
        """
        
        model = InceptionV3(weights=None, include_top=False)
        model.set_weights(model_weights.value)
        return model


    def preprocess_img(data):
        """
        Prend une liste représentative de nos images, les transforme
        en array et change le format de l'image. Retourne les images
        auxquelles on applique le preprocessing propre à notre modèle.
        """
        
        image = np.asarray(data)
        image = np.resize(image, (299, 299, 3))

        return preprocess_input(image)

    def featurize_series(model, content_series):
        """
        Récupère toutes nos images pour y appliquer notre preprocessing
        et on applique notre modèle, avant de faire une classification.
        Retourne une série composée de nos features.
        
                Parameters:
                    model : notre modèle (InceptionV3 ici)
                    content_series : nos données images
        """
        
        model_input = np.stack(content_series.map(preprocess_img)) 
        preds = model.predict(model_input)
        model_output = [x.flatten() for x in preds]
        return pd.Series(model_output)

    @pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
    def udf_featurize_series(content_series_iter):
        """
        UDF (User Defined Function) wrapper pour notre fonction de featurisation.
        Retourne une colonne de DataFrame Spark de type ArrayType(FloatType)
        
                Parameters:
                content_series_iter : série de nos données images
        """
        model = model_pretrained()
        for content_series in content_series_iter:
            yield featurize_series(model, content_series)

    ###########################################################################################
    ###########################################################################################
    ###########################################################################################

    # Préparation de nos données : 

    # On se connecte à notre bucket S3
# On se connecte à notre bucket S3
    session = boto3.session.Session(aws_access_key_id='XXX',
                                    aws_secret_access_key='XXX')

    s3_client = session.client(service_name='s3', region_name='eu-west-3')

    # Et on se prépare à lire nos fichiers
    prefix = 'Data'
    sub_folders = s3_client.list_objects_v2(Bucket='p8ocrvle', Prefix=prefix)

    # On lit les fichiers présent dans notre dossiers "Data"
    # On récupère l'emplacement des fichiers
    lst_path = []
    for key in sub_folders['Contents']:
        file = key['Key']
        file = file.replace(prefix + '/', '')
        lst_path.append('s3://p8ocrvle/Data/' + file)

    # On crée notre DataFrame et on y met l'emplacement de fichiers
    rdd = spark.sparkContext.parallelize(lst_path)
    row_rdd = rdd.map(lambda x: Row(x))
    df_pyspark = spark.createDataFrame(row_rdd, ['path'])

    print ('------ Notre dataframe a bien été créé ------')


    # On récupère nos labels
    df_pyspark = df_pyspark.withColumn('label', split(col('path'), '/').getItem(4))

    # On ajoute une colonne avec nos données images
    udf_image = udf(load_img, ArrayType(IntegerType()))
    df_pyspark = df_pyspark.withColumn('data', udf_image('path'))

    # On va utiliser un StringIndexer pour encoder nos labels
    stringIndexer = StringIndexer(inputCol='label', outputCol='label_encoded')
    sI = stringIndexer.fit(df_pyspark)

    # On encode et on convertit nos labels en Integer (lisibilité)
    image_df = sI.transform(df_pyspark)
    image_df = image_df.withColumn('label_encoded', col('label_encoded').cast(IntegerType()))

    # On réorganise nos colonnes (lisibilité)
    image_df = image_df.select('path', 'label', 'label_encoded', 'data')

    print ('------ Preprocessing prêt à être effectué ------')

    ###########################################################################################
    ###########################################################################################
    ###########################################################################################

    # Preprocessing : 

    # On charge notre model et les poids associés
    model = InceptionV3(include_top=False)
    model_weights = spark.sparkContext.broadcast(model.get_weights()) 

    # On applique notre preprocessing
    features_df = image_df.select(col('path'), col('label'), col('label_encoded'),
                                udf_featurize_series('data').alias('features'))

    print ('------ Preprocessing a bien été appliqué ------')

    ###########################################################################################
    ###########################################################################################
    ###########################################################################################

    # Standardisation des données avant une réduction de dimension

    # On transforme nos données en vecteurs
    list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
    features_df = features_df.select(col('path'),  col('label'),
                                    list_to_vector_udf(features_df['features']).alias('features_vect'))

    # On standardise nos données
    standardizer = StandardScaler(withMean=True, withStd=True,
                                inputCol='features_vect',
                                outputCol='feats_scaled')
    std = standardizer.fit(features_df)
    features_df_scaled = std.transform(features_df)

    print ('------ Nos données ont été standardisées ------')

    ###########################################################################################
    ###########################################################################################
    ###########################################################################################

    # PCA :

    pca = PCA(k=8, inputCol='feats_scaled', outputCol='pca')
    modelpca = pca.fit(features_df_scaled)
    transformed = modelpca.transform(features_df_scaled)

    print ('------ Notre PCA a bien été effectuée ------')

    ###########################################################################################
    ###########################################################################################
    ###########################################################################################

    # Enregistrement :

    # On enregistre dans un nouveau dossier dans notre bucket S3
    features_df_scaled.write.parquet(path='s3://p8ocrvle/FeaturesScript/', mode='overwrite')

    print ('------ Nos données ont bien été enregistrées ------')
