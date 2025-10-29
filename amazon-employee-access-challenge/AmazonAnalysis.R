########################################
## AMAZON EMPLOYEE ACCESS PREDICTIONS ##
########################################

# libraries
library(tidymodels)
library(tidyverse)
library(vroom)
library(glmnet)
library(embed)
library(janitor)
library(kknn)
library(discrim)
library(naivebayes)
library(keras)
library(tune)
library(tictoc)
library(kernlab)


library(tensorflow)


# read in data
train_data <- vroom("/Users/eliseclark/Documents/Fall 2025/Stat 348/AmazonEmployeeComp/amazon-employee-access-challenge/train.csv")
testData <- vroom("/Users/eliseclark/Documents/Fall 2025/Stat 348/AmazonEmployeeComp/amazon-employee-access-challenge/test.csv")

# clean up variable names
train_data <- clean_names(train_data)
testData <- clean_names(testData)

train_data <- train_data %>% 
  mutate(action = as.factor(action))


###### EDA ######
ggplot(data = train_data, aes(x = action)) + 
  geom_bar(fill = "plum3") + 
  xlab("Action") + ylab("Frequency") + 
  ggtitle("Amazon Employee Access Counts")


table(train_data$action, train_data$role_family)



###### BASE RECIPE ######

my_recipe <- recipe(action ~ . , data = train_data) %>%
  step_mutate_at(c(resource, mgr_id, role_rollup_1, role_rollup_2, role_deptname, 
                   role_title, role_family_desc, role_family, role_code), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <.1% 
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_nominal_predictors()) %>%
                   step_pca(all_predictors(), threshold = 0.85)


prep <- prep(my_recipe)
baked_data<- bake(prep, new_data = train_data)



###### LOGISTIC REGRESSION MODEL ######
logRegModel <- logistic_reg() %>% 
  set_engine("glm") 

## Put into a workflow here 
log_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(logRegModel) %>%
  fit(data = train_data)

## Make predictions
## with type="prob" amazon_predictions will have 2 columns
## one for Pr(0) and the other for Pr(1)!
## with type="class" it will just have one column (0 or 1)
amazon_predictions <- predict(log_workflow,
                              new_data = testData,
                              type = "prob") # "class" or "prob"
# drop 1st column 
amazon_preds <- amazon_predictions %>% 
  select(-.pred_0)

amazon_preds <- amazon_preds %>%
  mutate(id = row_number()) %>%
  select(id, .pred_1) %>%        
  rename(action = .pred_1)   


vroom_write(x = amazon_preds, file = "/Users/eliseclark/Documents/Fall 2025/Stat 348/AmazonEmployeeComp/pca1.csv", delim = ",")


###### PENALIZED LOGISTIC REGRESSION ######

# recipe with target coding 
penlog_recipe <- recipe(action ~ ., data = train_data) %>%
  step_mutate_at(c(resource, mgr_id, role_rollup_1, role_rollup_2, 
                   role_deptname, role_title, role_family_desc, 
                   role_family, role_code), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_lencode_glm(c(resource, mgr_id, role_rollup_1, role_rollup_2, 
                       role_deptname, role_title, role_family_desc, 
                       role_family, role_code), outcome = vars(action)) %>%
  step_rm(mgr_id) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_predictors())


#target encoding (must be 2-f8
# also step_lencode_glm() and step_lencode_bayes()

prep <- prep(penlog_recipe)
penlog_data <- bake(prep, new_data = train_data)

# penalized regression model 
penlog_mod <- logistic_reg(mixture = tune(), penalty = tune()) %>%
  set_engine("glmnet")

penlog_workflow <- workflow() %>%
  add_recipe(penlog_recipe) %>% add_model(penlog_mod) 

# Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                              mixture(),
                              levels = 4) ## L^2 total tuning possibilities
# Split data for CV
folds <- vfold_cv(train_data, v = 3, repeats = 2)

# Run the CV
CV_results <- penlog_workflow %>%
 tune_grid(resamples = folds,        # Mcallentexas2324!
            grid = tuning_grid,
            metrics = metric_set(roc_auc))


# Find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

# Finalize the workflow & fit it
final_wf <-penlog_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)

# predictions
penlog_preds <- final_wf %>%
  predict(new_data = testData, type = "prob")

penlog_preds <- penlog_preds %>%
  select(-.pred_0) %>%
  mutate(id = row_number()) %>%
  select(id, .pred_1) %>%        
  rename(action = .pred_1) 


# kaggle files
vroom_write(x = penlog_preds, file = "/Users/eliseclark/Documents/Fall 2025/Stat 348/AmazonEmployeeComp/pca2.csv", delim = ",")



###### CLASSIFICATION RANDOM FORESTS ######

# forest recipe
forest_recipe <- recipe(action ~ ., data = train_data) %>%
  step_mutate_at(c(resource, mgr_id, role_rollup_1, role_rollup_2, 
                   role_deptname, role_title, role_family_desc, 
                   role_family, role_code), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_lencode_glm(all_nominal_predictors(), outcome = vars(action)) %>%
  step_interact(terms = ~ resource:role_deptname + resource:role_title + mgr_id:role_family) %>%  # added interaction
  step_zv(all_predictors())


forest_prep <- prep(forest_recipe)
forest_data <- bake(forest_prep, new_data = train_data)


# model 
forest_lm <- rand_forest(mtry = tune(),
            min_n = tune(),
            trees = tune()) %>%
            set_engine("ranger") %>%
            set_mode("classification")


# workflow
forest_wf <- workflow() %>%
  add_recipe(forest_recipe) %>% add_model(forest_lm) 

# grid of tuning values
tuning_grid <- grid_random(
  mtry(range = c(1, floor(sqrt(ncol(train_data))))),
  min_n(range = c(2, 15)),
  trees(range = c(500, 1500)),
  size = 50)

# set up K-fold CV
folds <- vfold_cv(train_data, v = 5, repeats = 2)

tic()
CV_results <- forest_wf %>%
  tune_grid(
    resamples = folds,        
    grid = tuning_grid,
    metrics = metric_set(roc_auc),
    control = control_grid(verbose = TRUE))
toc()

# find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

# finalize workflow and predict
final_wf <- forest_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)

# predictions
forest_preds <- final_wf %>%
  predict(new_data = testData, type = "prob")

forest_preds <- forest_preds %>%
  mutate(id = row_number()) %>%
  select(id, .pred_1) %>%        
  rename(action = .pred_1)  

vroom_write(x = forest_preds, file = "/Users/eliseclark/Documents/Fall 2025/Stat 348/AmazonEmployeeComp/preds_uno.csv", delim = ",")



###### KNN MODEL ######

# recipe
knn_recipe <- recipe(action ~ ., data = train_data) %>%
  step_rm(mgr_id) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(action)) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_predictors())

# knn model
knn_model <- nearest_neighbor(neighbors = tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

# workflow 
knn_wf <- workflow() %>%
  add_recipe(knn_recipe) %>%
  add_model(knn_model)


# grid of tuning values
tuning_grid <- grid_regular(neighbors(range = c(1, 50)),
                            levels = 10)

# set up K-fold CV
folds <- vfold_cv(train_data, v = 4, repeats = 2)

CV_results <- knn_wf %>%
  tune_grid(resamples = folds,        
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

# finalize workflow and predict
final_wf <- knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)

# predictions
knn_preds <- final_wf %>%
  predict(new_data = testData, type = "prob")

knn_preds <- knn_preds %>%
  mutate(id = row_number()) %>%
  select(id, .pred_1) %>%        
  rename(action = .pred_1)  

vroom_write(x = knn_preds, file = "/Users/eliseclark/Documents/Fall 2025/Stat 348/AmazonEmployeeComp/knn_preds2.csv", delim = ",")



############ NAIVE BAYES ############

# recipe
nb_recipe <- recipe(action ~ ., data = train_data) %>%
  step_rm(mgr_id) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(action)) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())


## nb model
nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

# workflow 
nb_wf <- workflow() %>%
  add_recipe(nb_recipe) %>%
  add_model(nb_model)

# Tune smoothness and Laplace here
# grid of tuning values
tuning_grid <- grid_regular(Laplace(range = c(0,2)), 
                            smoothness(range = c(0,2)),
                            levels = 5)

# set up K-fold CV
folds <- vfold_cv(train_data, v = 3, repeats = 2)

CV_results <- nb_wf %>%
  tune_grid(resamples = folds,        
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

# finalize workflow and predict
final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)


# Predict
nb_preds <- final_wf %>%
  predict(new_data = testData, type = "prob")

nb_preds <- nb_preds %>%
  mutate(id = row_number()) %>%
  select(id, .pred_1) %>%        
  rename(action = .pred_1) 

vroom_write(x = nb_preds, file = "/Users/eliseclark/Documents/Fall 2025/Stat 348/AmazonEmployeeComp/nb_preds.csv", delim = ",")

# above 0.885


########## NEURAL NETWORK ##########
py_require("tensorflow")

nn_recipe <- recipe(action ~ ., data = train_data) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1)

nn_model <- mlp(hidden_units = tune(),
                epochs = 50) %>% 
  set_engine("keras") %>% 
  set_mode("classification")

nn_wf <- workflow() %>%
  add_model(nn_model) %>%
  add_recipe(nn_recipe)
                
nn_tuneGrid <- grid_regular(hidden_units(range = c(1, 50)),
                            levels = 2)


# set up K-fold CV
folds <- vfold_cv(train_data, v = 3, repeats = 2)

CV_results <- nn_wf %>%
  tune_grid(resamples = folds,        
            grid = nn_tuneGrid,
            metrics = metric_set(roc_auc))

# find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

# finalize workflow and predict
final_wf <- nn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)

# Predict
nn_preds <- final_wf %>%
  predict(new_data = testData, type = "prob")

nn_preds <- nn_preds %>%
  mutate(id = row_number()) %>%
  select(id, .pred_1) %>%        
  rename(action = .pred_1) 

vroom_write(x = nn_preds, file = "/Users/eliseclark/Documents/Fall 2025/Stat 348/AmazonEmployeeComp/nn_preds.csv", delim = ",")


################# SUPPORT VECTOR MACHINES ##################
### RADIAL MODEL INCLUDES SMOTE ###

svm_recipe <- recipe(action ~ ., data = train_data) %>%
  step_rm(mgr_id) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(action)) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())


svm_prep <- prep(svm_recipe)
svm_data <- bake(svm_prep, new_data = train_data)

## SVM models
svmPoly <- svm_poly(degree = tune(), 
                    cost = tune()) %>% # set or tune
                    set_mode("classification") %>%
                    set_engine("kernlab", maxiter = 5000)

# note that svmRadial worked best
svmRadial <- svm_rbf(rbf_sigma = tune(), cost = tune()) %>% # set or tune
              set_mode("classification") %>%
                set_engine("kernlab")

svmLinear <- svm_linear(cost=tune()) %>% # set or tune
                set_mode("classification") %>%
                  set_engine("kernlab")

# workflow 
svm_wf <- workflow() %>%
  add_recipe(svm_recipe) %>%
  add_model(svmLinear)


# grid of tuning values
tuning_grid <- grid_regular(
  cost(range = c(-2, 1), trans = log10_trans()),        
  levels = 4)

# set up K-fold CV
folds <- vfold_cv(train_data, v = 3)


CV_results <- svm_wf %>%
  tune_grid(
    resamples = folds,        
    grid = tuning_grid,
    metrics = metric_set(roc_auc),
    control = control_grid(verbose = TRUE))

# find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

# finalize workflow and predict
final_wf <- svm_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)


# predict
svm_preds <- final_wf %>%
  predict(new_data = testData, type = "prob")

svm_preds <- svm_preds %>%
  mutate(id = row_number()) %>%
  select(id, .pred_1) %>%        
  rename(action = .pred_1) 


vroom_write(x = svm_preds, file = "/Users/eliseclark/Documents/Fall 2025/Stat 348/AmazonEmployeeComp/svmLinear_preds.csv", delim = ",")




  




