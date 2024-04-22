#Ler os Dados

set.seed(1)

# Ler os dados
df <- read.csv("/Users/bennymanuel/Documents/BitcoinHeistData.csv", stringsAsFactors=FALSE, header=TRUE)
summary(df)

#Limpeza de Dados


# simplificar dados
data <- df[sample(nrow(df),14370, replace = FALSE),] 

# remover coluna de endereço
data$address <- NULL

# converter tipo de variável
data$year <- as.numeric(data$year)
data$day <- as.numeric(data$day)
data$length <- as.numeric(data$length)
data$weight <- as.numeric(data$weight)
data$count <- as.numeric(data$count)
data$looped <- as.numeric(data$looped)
data$neighbors <- as.numeric(data$neighbors)
data$income <- as.numeric(data$income)

# transformar variável y em binário 0, 1
data$label <- ifelse(data$label=="white", 0, 1)

# pré-visualizar dados
tail(data)

#Dividir em conjuntos de treinamento e teste


flag <- sort(sample(nrow(data),4790, replace = FALSE))
btctrain <- data[-flag,]
btctest <- data[flag,]

## valor de resposta verdadeiro para os dados de treinamento e teste
y1    <- btctrain$label;
y2    <- btctest$label;

#Análise Exploratória de Dados (EDA)


# Visualizar correlações entre variáveis
corr<-cor(data)
library(corrplot)
corrplot(corr, method="circle")

# Definir label como fator
data$label <- as.factor(data$label)

library(ggplot2)
# variável de resposta
ggplot(data.frame(data$label), aes(x=data$label)) +
  geom_bar(fill="#702963") + xlab("Label") + ylab("Frequência") + ggtitle("Distribuição da Variável de Resposta 'label'")

#Método de Boosting

library(gbm)
library(magrittr)
library(dplyr)

# criar busca em grade
hyper_grid <- expand.grid(
  learning_rate = c(0.3, 0.1, 0.05, 0.01, 0.005),
  RMSE = NA,
  trees = NA
)

# executar busca em grade
for(i in seq_len(nrow(hyper_grid))) {
  
  # ajustar gbm
  set.seed(123)  # para reprodutibilidade
  train_time <- system.time({
    m <- gbm(
      formula = label ~ .,
      data = btctrain,
      distribution = "bernoulli",
      n.trees = 5000, 
      shrinkage = hyper_grid$learning_rate[i], 
      cv.folds = 10 
    )
  })
  
  # adicionar SSE, árvores e tempo de treinamento aos resultados
  hyper_grid$RMSE[i]  <- sqrt(min(m$cv.error))
  hyper_grid$trees[i] <- which.min(m$cv.error)
  hyper_grid$Time[i]  <- train_time[["elapsed"]]
  
}

# resultados
arrange(hyper_grid, RMSE)



#GBM
gbm.btc1 <- gbm(label ~ .,data=btctrain,
                distribution = 'bernoulli',
                n.trees = 5000, 
                shrinkage = 0.05,
                cv.folds = 10)

## Encontrar o número ótimo estimado de iterações
perf_gbm1 <- gbm.perf(gbm.btc1, method="cv") 
message("O número ótimo de iterações n.trees é: ", perf_gbm1)


# busca em grade
hyper_grid <- expand.grid(
  n.trees = perf_gbm1,
  shrinkage = 0.05,
  interaction.depth = c(3, 5, 7)
)

# criar função de ajuste do modelo
model_fit <- function(n.trees, shrinkage, interaction.depth) {
  set.seed(123)
  m <- gbm(
    formula = label ~ .,
    data = btctrain,
    distribution = "bernoulli",
    n.trees = n.trees,
    shrinkage = shrinkage,
    interaction.depth = interaction.depth,
    cv.folds = 10
  )
  # calcular RMSE
  sqrt(min(m$cv.error))
}

# realizar busca em grade com programação funcional
hyper_grid$rmse <- purrr::pmap_dbl(
  hyper_grid,
  ~ model_fit(
    n.trees = ..1,
    shrinkage = ..2,
    interaction.depth = ..3
  )
)

# resultados
arrange(hyper_grid, rmse)


# atualizar modelo
gbm.btc2 <- gbm(label ~ .,data=btctrain,
                distribution = 'bernoulli',
                n.trees = perf_gbm1, 
                shrinkage = 0.05, 
                interaction.depth = 5,
                cv.folds = 10)

## Quais variâncias são importantes
summary(gbm.btc2)


## Erro de treinamento
message("Probabilidades de classificação previstas das primeiras dez linhas:")
pred1gbm <- predict(gbm.btc2,newdata = btctrain, n.trees=perf_gbm1, type="response")
pred1gbm[1:10]

message("Valores de label previstos das primeiras dez linhas: ")
y1hat <- ifelse(pred1gbm < 0.5, 0, 1)
y1hat[1:10]

message("O erro de treinamento é: ", sum(y1hat != y1)/length(y1))


## Erro de Teste
y2hat <- ifelse(predict(gbm.btc2, newdata = btctest[,-9], n.trees=perf_gbm1, type="response") < 0.5, 0, 1)
message("O erro de teste é: ", mean(y2hat != y2) )

boost <- mean(y2hat != y2)

#Random Forest


library(randomForest)
library(caret)


# parâmetros ajustados
mtry_tune = round(sqrt(8), 0)
nodesize_tune = 1
ntree_tune = 500

control <- trainControl(method = 'repeatedcv',number = 5)

storeMaxtrees <- list()
tuneGrid <- expand.grid(.mtry = mtry_tune)
for (ntree in c(500, 1000, 2000, 5000)) {
  set.seed(1)
  rf.maxtrees <- train(as.factor(label) ~ .,
                       data = btctrain,
                       method = "rf",
                       metric = "Accuracy",
                       tuneGrid = tuneGrid,
                       trControl = control,
                       importance = TRUE,
                       nodesize = nodesize_tune,
                       ntree = ntree)
  key <- toString(ntree)
  storeMaxtrees[[key]] <- rf.maxtrees
}
resultsTree <- resamples(storeMaxtrees)
res = summary(resultsTree)
print(res)
ntree_tune <- res$models[which.max(res$statistics$Accuracy[,"Mean"])]


message("O valor ajustado de mtry é: ", mtry_tune)
message("O valor ajustado de nodesize é: ", nodesize_tune)
message("O valor ajustado de ntree é: ", ntree_tune)


#F: Random Forest
modF <- randomForest(as.factor(label) ~., data=btctrain, 
                     mtry = 3,
                     nodesize = 1,
                     ntree = 500,
                     importance=TRUE)

# verificar importância
importance(modF, type=1)
importance(modF, type=2)
varImpPlot(modF)

# erros de previsão
y2hatF = predict(modF, btctest, type='class')
message("O erro de teste previsto é: ", mean(y2hatF != y2))
rftest <- mean(y2hatF != y2)

#Métodos de Base

#A. Regressão Logística 
modA <- step(glm(label ~ ., data = btctrain, family = "binomial"), trace=0)
summary(modA)

y2hatA <- ifelse(predict(modA, btctest[,-58], type="response" ) < 0.5, 0, 1)
message("O erro de teste é: ", sum(y2hatA != y2)/length(y2))
steplog <- mean( y2hatA != y2)


#B. Análise Discriminante Linear
library(MASS)
modB <- lda(btctrain[,1:8], btctrain[,9])
print(modB)


y2hatB <- predict(modB, btctest[,-9])$class
message("O erro de teste é: ", mean( y2hatB  != y2))
ldiscrim <-  mean( y2hatB != y2)


## C. Naive Bayes (com X completo)
library(e1071)
modC <- naiveBayes(as.factor(label) ~. , data = btctrain)
summary(modC)

y2hatC <- predict(modC, newdata = btctest, type="class")

message("O erro de teste é: ", mean( y2hatC != y2))
nbayes <- mean( y2hatC != y2)


#E: uma Árvore Simples
library(rpart)
modE0 <- rpart(label ~ .,data=btctrain, method="class", 
               parms=list(split="gini"))

# ajustar parâmetros
opt <- which.min(modE0$cptable[, "xerror"]); 
cp1 <- modE0$cptable[opt, "CP"];
modE <- prune(modE0,cp=cp1);
summary(modE)

y2hatE <-  predict(modE, btctest[,-9],type="class")
message("O erro de teste é: ", mean(y2hatE != y2))
singletree <- mean(y2hatE != y2)

#Resultados


# Imprimir todos os erros de treinamento
message("Erro de Teste de Cada Modelo: ")
testing_errors = c(boost, rftest, steplog, ldiscrim, nbayes, singletree)
models <- c("Boosting", "Random Forest", "Regressão Logística Sequencial", "Análise Discriminante Linear", "Naive Bayes", "Árvore Simples")
results_table <- data.frame(model=models, testing_error=testing_errors)

print(results_table)
