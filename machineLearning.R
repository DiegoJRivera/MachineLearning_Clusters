# Ejemplo Clustering Machine Learning 
# por: Diego Rivera

library(ggplot2)
library(tidyverse)
library(rpart)
install.packages('rpart.plot')
library(rpart.plot)
library(caret)
library(randomForest)
library(cluster)
library(xgboost)
library(gbm)

# Contexto
# Una compania de agricultura contrata sus servicios para analizar data de variedades de trigo. 
# En particular, la compania le dio datos con respecto a tres tipos de trigo: Kama, Rosa, y Canadian. 
# Existen siete variables geometricas para caracterizar estos tipos de trigo (valores continuos)
# cargue la data segun las instrucciones de abajo y responda las preguntas

# Seteando directorio de trabajo
setwd("~/GoogleDrive/UAI/1_MDS2019/7_MachineLearning")

url<-"https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
semillas <- read.table(url,header = F)
# Nombre de las variables
semillas.names=c("Area", "Perimetro", "Compacidad", "Largo del nucleo", "Ancho del nucleo",
             "Coeficiente de asimetria", "Largo ranura del nucleo", "Tipo de trigo")
colnames(semillas)=semillas.names
tiposemilla<-semillas$`Tipo de trigo` #etiqueta numerica del tipo de trigo 
semillas$`Tipo de trigo`<-as.factor(ifelse(semillas$`Tipo de trigo`==1,"Kama",
                                           ifelse(semillas$`Tipo de trigo`==2,"Rosa","Canadian"))) #etiqueta con el nombre del tipo de trigo

str(semillas)
summary(semillas)
names(semillas)

#######################################################################################################
# 1. Omita la variable Tipo de trigo, luego proponga un numero de clusters para el dataset via kmeans #
#######################################################################################################

# omitiendo variable tipo de trigo
semillas2 <- semillas[,-c(8)]

str(semillas2)
summary(semillas2)
names(semillas2)

# escalando data frama
semillas2 <- scale(semillas2)
summary(semillas2)

# proponiendo un numero de clusters (k) optimo para dataset semillas, por medio del metodo del "codo"
# Este metodo ve el porcentaje de la varianza explicada por cada grupo. Puede esperar que la variabilidad 
# aumente con el número de grupos, alternativamente, la heterogeneidad disminuye. La idea es encontrar el k que 
# está más allá de los rendimientos decrecientes. Agregar un nuevo clúster no mejora la variabilidad en los datos porque
# queda muy poca información para explicar.

# Acontinuacion se encontrará este punto usando la medida de heterogeneidad. La suma total de cuadrados dentro de los
# clústeres es tot.withinss en la lista devuelta por kmean ().

# Se graficará el codo y encontrará el numero de clusters óptimo de la siguiente manera:
  
# Se definira una función para calcular el total dentro de la suma de cuadrados de los clústeres
set.seed(6666) # fijando una semilla para homologar resultados a futuro
kmean_withinss <- function(k) {
  cluster <- kmeans(semillas2, k)
  return (cluster$tot.withinss)
}
# function(k) -> establece el número de argumentos en la función
# kmeans -> ejecuta el algoritmo k veces
# return (cluster$tot.withinss) -> almacena el total dentro de la suma de cuadrados de los clusters

# Ejecutando los tiempos del algoritmo. Seteando cluster maximo 
max_k <- 20 

# Ejecuta el algoritmo en un rango de k. Ejecuta la función kmean_withinss() en un rango 2: max_k, es decir, de 2 a 30
wss <- sapply(2:max_k, kmean_withinss)

# Creaando un data frame con los resultados del algoritmo
# Luego de crear y probar la función kmean_withinss(), se puede ejecutar el algoritmo k-mean en un rango de 2 a 30,
# y almacenar los valores tot.withinss.
# Creando data frame "codo" para graficar
codo <- data.frame(2:max_k, wss)

# Visualizando los resultados obtenidos
ggplot(codo, aes(x = X2.max_k, y = wss)) +
  geom_point() +
  geom_line() +
  scale_x_continuous(breaks = seq(1, max_k, by = 1))

# gráficamente aprecio que el k óptimo es 5, donde la curva comienza a tener un rendimiento decreciente.

###############################################################################
# 2.En base al resultado anterior, corra kmeans con el valor de k encontrado  #
# (use una semilla antes de correr kmeans set.seed(XXXX)). En que cluster se  #
# tiene el promedio de Perimetro mas grande. Para ese cluster identificado,   #
# como se distribuye el tipo de trigo.                                        #
###############################################################################

set.seed(1986)
respuesta2 <- kmeans(semillas[,-c(8)], 5, nstart=10)
respuesta2

# Calcular la media de cada variable por grupos utilizando los datos originales:
aggregate(semillas[,-c(8)], by=list(cluster=respuesta2$cluster), mean)

max(respuesta2$centers[,2])
# 16.46917, cluster 5

# Distribucion del tipo de trigo en el cluster de mayor promedio de perimetro de grano
table(semillas$`Tipo de trigo`[respuesta2$cluster == 5])
# Canadian     Kama     Rosa 
#        0        0       48 

# visualización de los clusters
clusplot(semillas2, respuesta2$cluster, color=TRUE, shade=TRUE, labels=0, lines=0)

#############################################################################################################################
# 3.Considere el dataset completo. Entrene un arbol de clasificacion para clasificar automaticamente el tipo de trigo segun #
# los valores de las 7 variables geometricas de las semillas.                                                               #
# para esto genere el siguiente conjunto de entrenamiento y prueba                                                          #
#############################################################################################################################

set.seed(123)
sub <- sample(nrow(semillas), floor(nrow(semillas) * 0.7))
train<-data.frame(semillas[sub, ]) #conjunto de entrenamiento 70%
test<-data.frame(semillas[-sub, ]) #conjunto de prueba 30%

c(train, test)
head(test)
###########################################################################################
# Entregue la matriz de confusion al evaluar en el conjunto de prueba y senale cual       #
# es el porcentaje de clasificaciones correctas                                           #
# Indique para que tipo de trigo, el clasificador es mas efectivo                         #
# Considere para esta pregunta la construccion del arbol tanto usando Gini como entropia. #
# Señale con cual indice obtiene mejores resultados en clasificacion.                     #
###########################################################################################

# Entrenando el modelo

###########
# By Gini #
###########
arbol_1 <- rpart(train$Tipo.de.trigo ~ ., data = train, method = "class")

# Evaluando el modelo por Gini
arbol_1

# visualizando el modelo arbol 1
rpart.plot(arbol_1)

# Cada uno de los rectángulos del grafico representa un nodo del arbol_1, con su regla de clasificación.
# Cada nodo está coloreado de acuerdo a la categoría mayoritaria entre los datos que agrupa.
# Esta es la categoría que ha predicho el modelo para ese grupo.

# Dentro del rectángulo de cada nodo se nos muestra qué proporción de casos pertenecen a cada categoría y la proporción del 
# total de datos que han sido agrupados allí.
# Por ejemplo, el rectángulo en el extremo inferior izquierdo de la gráfica tiene 94% de casos en el tipo 1 (Kama), 
# y 6% del tipo 2 (Rosa) y 3 0% del tipo 3 (Canadian), que representan 35% de todos los datos.

# Generar un vector con los valores predichos por el modelo entrenado
predictions <- predict(arbol_1, test, type="class")
predictions

# Cruzando la predicción con los datos reales del set de prueba para generar la matriz de confusión para modelo gini
confusionMatrix(predictions, test$Tipo.de.trigo)
# Accuracy : 0.9048          

# Reference
# Prediction Canadian Kama Rosa
# Canadian         18    3    0
# Kama              1   20    2
# Rosa              0    0   19

# La mejor prediccion espara el grano tipo Rosa, con 19 clasificaciones correctas y 0 errores. A diferencia de Kama con 20
# aciertos, pero 3 errores y Canadia con 18 aciertos y 3 errores.

###############
# By Entropia #
###############
arbol_2 <- rpart(train$Tipo.de.trigo ~ ., data = train, method = "class",
                parms = list(split = "information"))

# Evaluando el modelo por Entropia
arbol_2

# visualizando el modelo arbol 2
rpart.plot(arbol_2)

# Generar un vector con los valores predichos por el modelo por entropia entrenado
prediction2 <- predict(arbol_2, test, type="class")
prediction2

# Cruzando la predicción con los datos reales del set de prueba para generar la matriz de confusión para modelo entropia
confusionMatrix(prediction2, test$Tipo.de.trigo)
# Accuracy : 0.873  

# Reference
# Prediction Canadian Kama Rosa
# Canadian         19    6    0
# Kama              0   17    2
# Rosa              0    0   19

# Nuevamente la mejor prediccion espara el grano tipo Rosa, con 19 clasificaciones correctas y 0 errores. A diferencia de Kama, 
# esta vez con 17 aciertos y 2 errores y Canadia con 19 aciertos y 6 errores.

####################################################################################
# 4.Usando el mejor arbol resultante de la pregunta anterior. Explique en palabras #
# las reglas para clasificar un trigo del tipo Canadian.                           #
####################################################################################
arbol_1
# En base al mejor resultado obtenido, arbol_1 (gini con un Accuracy : 0.9048), las reglas para clasificar un trigo de tipo
# Canadian se basan en dos variables: Largo.ranura.del.nucleo y Area. Por lo tanto el grano deberá tener un Area < 13.41 y
# Largo.ranura.del.nucleo< 5.615 y Largo.ranura.del.nucleo>=4.8265

#################################################################################################################
# 5.Usando la misma particion de train y test. Use train para entrenar un random forest. Proponga algun enfoque #
# usando solo el conjunto train para determinar valores apropiados para los dos hiperparametros que             #
# deben ser definidos por el usario. Luego evalue el modelo usando el conjunto test y senale cual               #
# es el porcentaje de clasificaciones correctas. Entegue la visualizacion del ranking de importancia de los     #
# atributos. Concluya, cuales son los top 2 atributos para este problema de clasificacion.                      #
#################################################################################################################

# Ajustando modelo random forest
modeloRandomForest <- randomForest(train$Tipo.de.trigo ~ ., data = train, importance = TRUE)

# resumen del modelo random forest
modeloRandomForest

# prediciendo tipo de grano
prediction3 <- predict(modeloRandomForest, test, type = "class")
prediction3

# Comprobación de precisión de clasificación. idem resultado matriz de confusion
table(prediction3, test$Tipo.de.trigo)

# Cruzando la predicción con los datos reales del set de prueba para generar la matriz de confusión para modelo random forest
confusionMatrix(prediction3, test$Tipo.de.trigo)
# Accuracy : 0.9206

# Reference
# Prediction Canadian Kama Rosa
# Canadian       19    2    0
# Kama            0   21    3
# Rosa            0    0   18

# importancia de los valores
modeloRandomForest$importance

# graficando la importancia de las variables (valores mas altos indican mayor importancia)
varImpPlot(modeloRandomForest,type=2)

# Se concluye que de acuerdo al criterio de MeanDecreaseGini, las variables mas imortantes son:
# Largo.ranura.del.nucleo (19.754604) y Perimetro (18.860392)

# Ajustando segundo modelo random forest
modeloRandomForest2 <- randomForest(train$Tipo.de.trigo ~ ., data = train, importance = TRUE, ntree = 4000, mtry=2)

# prediciendo tipo de grano
prediction4 <- predict(modeloRandomForest2, test, type = "class")

# Cruzando la predicción con los datos reales del set de prueba para generar la matriz de confusión para modelo random forest2
confusionMatrix(prediction4, test$Tipo.de.trigo)

# importancia de los valores
modeloRandomForest2$importance

# graficando la importancia de las variables (valores mas altos indican mayor importancia)
varImpPlot(modeloRandomForest2,type=2)

# mismo resultados de accuracy del modelo anterior de random forest, con importancia de variables distintas.
# Perimetro (20.370073) y Area (19.881856)

###########################################################################################################################
# 6. Entrene un clasificador gradient boosting machine. Se??ale cual es el mejor numero de arboles para considerar en     #
# el ensamble. Entregue el ranking de las variables segun su nivel de importancia para clasificar.Luego evalue el         #
# modelo usando el conjunto test y senale cual es el porcentaje de clasificaciones correctas. Considerando los resultdos  #
# de clasificacion obtenidos en la pregunta 3, 4 y 5, concluya cual es el mejor modelo para clasificar para este dataset. #
###########################################################################################################################
set.seed(666)
traingbm <- train
testgbm <- test

gbm.model_1<-gbm(Tipo.de.trigo ~ .,distribution="multinomial",
                data=traingbm,n.trees=10000,shrinkage = 0.01,cv.folds=5)
print(gbm.model_1)

# importancia de las variables.
summary(gbm.model_1)

# determinando el numero optimo de arboles via validacion cruzada
arboles_optimo <- gbm.perf(gbm.model_1, method="cv")
arboles_optimo # 575 optimo de arboles para considerar en el ensamble.

# prediciendo tipo de grano
predictionsgbm <- predict(gbm.model_1, testgbm, n.trees=arboles_optimo, type="response")

predictionsgbm <- predict.gbm(gbm.model_1, testgbm, n.trees=arboles_optimo, type="response")

predictionsgbm <-colnames(predictionsgbm)[apply(predictionsgbm, 1, which.max)]

predictionsgbm <- as.factor(predictionsgbm)

# matriz de confusión para modelo gradient boosting machine
confusionMatrix(predictionsgbm,testgbm$Tipo.de.trigo)

# Accuracy : 0.9365    

# Reference
# Prediction Canadian Kama Rosa
# Canadian         19    2    0
# Kama              0   21    2
# Rosa              0    0   19

# computo medidas de desempeño
postResample(pred = predictionsgbm, obs = test$Tipo.de.trigo)

# Final el mejor modelo encontrado en todo el desarrollo fue el gradient boosting machine, con un Accuracy : 0.9365.

# Cin una prediccion para el grano tipo Rosa de 19 clasificaciones correctas y 0 errores. Para el grano Kama con 21
# aciertos y 2 errores y Canadia con 19 aciertos y 2 errores.

#Entrega: v??a webcursos. Responda en este mismo archivo pero cambie el nombre del archivo a su nombre y apellido.
#Se habilitar?? un link que estar?? disponible para subir el archivo .R, el link estar?? habilitado hasta 
#el viernes 24 de enero hasta las 23:55 horas.