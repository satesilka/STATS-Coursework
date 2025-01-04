# Load necessary libraries
library(ggplot2)
library(MASS)
library(ggfortify)
library(factoextra)
library(heatmap3)
library(randomForest)

# Load datasets
data_elements <- read.csv("dataset2.csv")
data_components <- read.csv("dataset1.csv")

# PCA for elements
elements_pca <- prcomp(data_elements[, 4:28], center = TRUE, scale. = TRUE)
summary(elements_pca)
plot(elements_pca)
biplot(elements_pca, cex = 0.5)

# PCA loadings for elements
sorted_pc1_loadings_elements <- sort(elements_pca$rotation[, 1], decreasing = TRUE)
print(sorted_pc1_loadings_elements)
elements_pca_scores <- elements_pca$x[, 1:8]

# LDA for elements
lda_model_elements <- lda(elements_pca_scores, grouping = data_elements$Months, CV = TRUE)
print(lda_model_elements)
confusion_matrix_elements <- table(real = data_elements$Months, predicted = lda_model_elements$class)
print(confusion_matrix_elements)
class_wise_accuracy <- diag(prop.table(confusion_matrix_elements, 1))
print(class_wise_accuracy)
overall_accuracy <- sum(diag(prop.table(confusion_matrix_elements)))
print(overall_accuracy)

# LDA plot for elements
lda_predictions <- predict(lda(elements_pca_scores, grouping = data_elements$Months))
lda_plot <- as.data.frame(lda_predictions$x)
lda_plot$Group <- data_elements$Months

ggplot(lda_plot, aes(x = LD1, y = LD2, color = Group)) +
  geom_point(size = 3) +
  labs(title = "LDA on Elements Data", x = "LD1", y = "LD2") +
  theme_minimal()

# PCA for components
components_pca <- prcomp(data_components[, 4:26], scale. = TRUE)
summary(components_pca)
plot(components_pca)
biplot(components_pca, cex = 0.5)

# PCA loadings for components
sorted_pc1_loadings_components <- sort(components_pca$rotation[, 1], decreasing = TRUE)
print(sorted_pc1_loadings_components)
sorted_pc2_loadings_components <- sort(components_pca$rotation[, 2], decreasing = TRUE)
print(sorted_pc2_loadings_components)

components_pca_scores <- components_pca$x[, 1:7]

# LDA for components
lda_model_components <- lda(components_pca_scores, grouping = data_components$Months, CV = TRUE)
print(lda_model_components)
confusion_matrix_components <- table(real = data_components$Months, predicted = lda_model_components$class)
print(confusion_matrix_components)
class_wise_accuracy_components <- diag(prop.table(confusion_matrix_components, 1))
print(class_wise_accuracy_components)
overall_accuracy_components <- sum(diag(prop.table(confusion_matrix_components)))
print(overall_accuracy_components)

# LDA plot for components
lda_predictions_components <- predict(lda(components_pca_scores, grouping = data_components$Months))
lda_plot_components <- as.data.frame(lda_predictions_components$x)
lda_plot_components$Group <- data_components$Months

ggplot(lda_plot_components, aes(x = LD1, y = LD2, color = Group)) +
  geom_point(size = 3) +
  labs(title = "LDA on Components Data", x = "LD1", y = "LD2") +
  theme_minimal()

# PCA Scores by Months and Samples
data_elements$Samples <- factor(data_elements$Samples, levels = c("Frozen", "Shade"))
data_components$Samples <- factor(data_components$Samples, levels = c("Frozen", "Shade"))

# Plot PCA scores for elements and components
elements_pca_scores <- as.data.frame(elements_pca$x)
elements_pca_scores$Samples <- data_elements$Samples
elements_pca_scores$Months <- data_elements$Months

ggplot(elements_pca_scores, aes(x = PC1, y = PC2, color = Months)) +
  geom_point(size = 3, aes(shape = Samples)) +
  scale_shape_manual(values = c(15, 17)) +
  labs(title = "Elements PCA Scores by Months", x = "PC1", y = "PC2") +
  theme_minimal()

components_pca_scores <- as.data.frame(components_pca$x)
components_pca_scores$Samples <- data_components$Samples
components_pca_scores$Months <- data_components$Months

ggplot(components_pca_scores, aes(x = PC1, y = PC2, color = Months)) +
  geom_point(size = 3, aes(shape = Samples)) +
  scale_shape_manual(values = c(15, 17)) +
  labs(title = "Components PCA Scores by Months", x = "PC1", y = "PC2") +
  theme_minimal()

# Hierarchical clustering
hcl <- hclust(dist(data_components[, 4:26]), method = "complete")
plot(hcl, labels = data_components$Samples, cex = 0.6, main = "Cluster Dendrogram")
rect.hclust(hcl, 2)

# Heatmaps for correlation matrices
heatmap3(cor(data_elements[data_elements$Samples == "Frozen", 4:28]),
         main = "Frozen Samples - Elements")
heatmap3(cor(data_elements[data_elements$Samples == "Shade", 4:28]),
         main = "Shade-Dried Samples - Elements")

heatmap3(cor(data_components[data_components$Samples == "Frozen", 4:26]),
         main = "Frozen Samples - Components")
heatmap3(cor(data_components[data_components$Samples == "Shade", 4:26]),
         main = "Shade-Dried Samples - Components")

frozen_elements_vs_components <- cor(data_elements[data_elements$Samples == "Frozen", 4:28], 
                                     data_components[data_components$Samples == "Frozen", 4:26])
heatmap3(frozen_elements_vs_components, scale = "none", revC = TRUE,
         main = "Frozen Samples - Elements vs Components")

shade_elements_vs_components <- cor(data_elements[data_elements$Samples == "Shade", 4:28], 
                                    data_components[data_components$Samples == "Shade", 4:26])
heatmap3(shade_elements_vs_components, scale = "none", revC = TRUE,
         main = "Shade-Dried Samples - Elements vs Components")

# Impact of volcanic ash on samples

# Subset data into ash-exposed and non-ash-exposed samples
no_ash_subset_elements <- data_elements[1:12, 4:28]
ash_subset_elements <- data_elements[13:36, 4:28]

no_ash_subset_components <- data_components[1:12, 4:26]
ash_subset_components <- data_components[13:36, 4:26]

# Perform PCA for elements and components
no_ash_pca <- prcomp(no_ash_subset_elements, scale. = TRUE)
ash_pca <- prcomp(ash_subset_elements, scale. = TRUE)

no_ash_pca_components <- prcomp(no_ash_subset_components, scale. = TRUE)
ash_pca_components <- prcomp(ash_subset_components, scale. = TRUE)

# Prepare data by removing unwanted columns
no_ash_elements <- no_ash_subset_elements[, !(colnames(no_ash_subset_elements) %in% c("Months", "Samples", "Replicate"))]
ash_elements <- ash_subset_elements[, !(colnames(ash_subset_elements) %in% c("Months", "Samples", "Replicate", "Cluster"))]
combined_elements <- rbind(no_ash_subset_elements, ash_subset_elements)

# Add an 'Exposure' column for elements
combined_elements$Exposure <- factor(c(rep("No Ash", nrow(no_ash_subset_elements)), 
                                       rep("Ash", nrow(ash_subset_elements))))

# Random Forest for elements data
set.seed(123)
rf_model_elements <- randomForest(Exposure ~ ., data = combined_elements)
print(rf_model_elements)
importance(rf_model_elements)
varImpPlot(rf_model_elements)

# Prepare data by removing unwanted columns for components
no_ash_components <- no_ash_subset_components[, !(colnames(no_ash_subset_components) %in% c("Months", "Samples", "Replicate"))]
ash_components <- ash_subset_components[, !(colnames(ash_subset_components) %in% c("Months", "Samples", "Replicate"))]
combined_components <- rbind(no_ash_subset_components, ash_subset_components)

# Add an 'Exposure' column for components
combined_components$Exposure <- factor(c(rep("No Ash", nrow(no_ash_subset_components)), 
                                         rep("Ash", nrow(ash_subset_components))))

# Random Forest for components data
set.seed(123)
rf_model_components <- randomForest(Exposure ~ ., data = combined_components)
print(rf_model_components)
importance(rf_model_components)
varImpPlot(rf_model_components)

# Summary of components and elements
summary(ash_subset_components)
summary(no_ash_subset_components)

summary(ash_subset_elements)
summary(no_ash_subset_elements)
