library(dplyr)
library(ggpubr)
library(reshape)
library(Rmisc)
library(ggpubr)
library(ggplot2)
library(reshape2)
library(plyr)

########### Import data ########################

setwd("/home/manvi/Documents/ML18-ensemble/results/new-results-21jan/compas/non-reweighed")

accuracy <- read.csv("accuracy.csv")
advDebias <- read.csv("adversarial.csv")
ensemble <- read.csv("ensemble.csv")
neural <- read.csv("neural_net.csv")
prej <- read.csv("prejudice.csv")

setwd("/home/manvi/Documents/ML18-ensemble/results/new-results-21jan/compas/reweighed")

accuracyRew <- read.csv("accuracy.csv")
advDebiasRew <- read.csv("adversarial.csv")
ensembleRew <- read.csv("ensemble.csv")
neuralRew <- read.csv("neural_net.csv")
prejRew <- read.csv("prejudice.csv")

################# Helper functions ##############

fmt_dcimals <- function(decimals=0){
  # return a function responpsible for formatting the 
  # axis labels with a given number of decimals 
  function(x) as.character(round(x,decimals))
}

########### Accuracy ###############
accuracy$X <- NULL
accuracyRew$X <- NULL
accuracy_nonrew_long <- stack(accuracy)
accuracy_nonrew_long['condition'] <- "non-reweighed"
accuracy_rew_long <- stack(accuracyRew)
accuracy_rew_long['condition'] <- "reweighed"

accuracy <- rbind(accuracy_nonrew_long, accuracy_rew_long)
accuracy$values <- accuracy$values * 100

accuracy_summary <- summarySE(accuracy, measurevar = "values", groupvars = c("condition", "ind"))

nonrew_df <- accuracy[accuracy$condition == "non-reweighed", ]
rew_df <- accuracy[accuracy$condition == "reweighed", ]

accuracy_plot <- ggplot(accuracy_summary, aes(x=factor(ind), y=values, fill=condition)) + 
  geom_bar(position=position_dodge(0.9), stat="identity",
           colour="black", # Use black outlines,
           size=.3) +      # Thinner lines
  scale_y_continuous(expand=c(0,0),breaks = seq(0, 87, by=5), limits = c(0,87)) +
  scale_x_discrete(labels = c("Adversarial\nDebiasing", "Multilayer\nPerceptron", "Prejudice\nRemover","Ensemble\nClassifier")) +
  geom_point(data=rew_df, position = position_nudge(x=0.22), size=3, alpha=0.6, shape=18, show.legend = FALSE) +
  geom_point(data=nonrew_df, position = position_nudge(x=-0.22), size=3, alpha=0.6, shape=18, show.legend = FALSE) +
  theme(legend.position = c(0.82,0.9),
        axis.ticks.x = element_blank(),
        axis.title.x = element_text(margin = margin(t = 10, r = 0, b = 0, l = 0))) +
  labs(x="Algorithm", y="Accuracy")

############ Mean Difference ######################

mean_diff_nonrew <- data.frame("advDebias" = advDebias$Mean.Difference,
                        "neural" = neural$Mean.Difference,
                        "prej" = prej$Mean.Difference,
                        "ensemble" = ensemble$Mean.Difference)
mean_diff_rew <- data.frame("advDebias" = advDebiasRew$Mean.Difference,
                               "neural" = neuralRew$Mean.Difference,
                               "prej" = prejRew$Mean.Difference,
                               "ensemble" = ensembleRew$Mean.Difference)
mean_diff_nonrew_long <- stack(mean_diff_nonrew)
mean_diff_nonrew_long['condition'] <- "non-reweighed"
mean_diff_rew_long <- stack(mean_diff_rew)
mean_diff_rew_long['condition'] <- "reweighed"

mean_diff <- rbind(mean_diff_nonrew_long, mean_diff_rew_long)

mean_diff_summary <- summarySE(mean_diff, measurevar = "values", groupvars = c("condition", "ind"))

nonrew_df <- mean_diff[mean_diff$condition == "non-reweighed", ]
rew_df <- mean_diff[mean_diff$condition == "reweighed", ]

mean_diff_plot <- ggplot(mean_diff_summary, aes(x=factor(ind), y=values, fill=condition)) + 
  geom_rect(aes(xmin = -Inf, xmax = Inf, ymin = -0.1, ymax = 0.1),
            fill = "palegreen2", alpha = 0.2,color = NA) +
  geom_hline(yintercept=0, 
             color = "darkgreen", size=1.5) +
  geom_bar(position=position_dodge(0.9), stat="identity",
           colour="black", # Use black outlines,
           size=.3) +      # Thinner lines
  scale_y_continuous(breaks = seq(-0.6, 0.4, by=0.1), limits = c(-0.6, 0.4), labels=fmt_dcimals(1)) +
  scale_x_discrete(labels = c("Adversarial\nDebiasing", "Multilayer\nPerceptron", "Prejudice\nRemover","Ensemble\nClassifier")) +
  geom_point(data=rew_df, position = position_nudge(x=0.22), size=3, alpha=0.6, shape=18, show.legend = FALSE) +
  geom_point(data=nonrew_df, position = position_nudge(x=-0.22), size=3, alpha=0.6, shape=18, show.legend = FALSE) +
  theme(legend.position = c(0.82,0.9),
        axis.ticks.x = element_blank(),
        axis.title.x = element_text(margin = margin(t = 10, r = 0, b = 0, l = 0))) +
  labs(x="Algorithm", y="Mean Difference")
  
############## Equal Opportunity Difference #####################

eqopp_nonrew <- data.frame("advDebias" = advDebias$Equal.Opportunity.Difference,
                               "neural" = neural$Equal.Opportunity.Difference,
                               "prej" = prej$Equal.Opportunity.Difference,
                               "ensemble" = ensemble$Equal.Opportunity.Difference)
eqopp_rew <- data.frame("advDebias" = advDebiasRew$Equal.Opportunity.Difference,
                            "neural" = neuralRew$Equal.Opportunity.Difference,
                            "prej" = prejRew$Equal.Opportunity.Difference,
                            "ensemble" = ensembleRew$Equal.Opportunity.Difference)
eqopp_nonrew_long <- stack(eqopp_nonrew)
eqopp_nonrew_long['condition'] <- "non-reweighed"
eqopp_rew_long <- stack(eqopp_rew)
eqopp_rew_long['condition'] <- "reweighed"

eqopp <- rbind(eqopp_nonrew_long, eqopp_rew_long)

eqopp_summary <- summarySE(eqopp, measurevar = "values", groupvars = c("condition", "ind"))

eqopp_plot <- ggplot(eqopp_summary, aes(x=factor(ind), y=values, fill=condition)) +
  geom_rect(aes(xmin = -Inf, xmax = Inf, ymin = -0.1, ymax = 0.1),
            fill = "palegreen2", alpha = 0.2,color = NA) +
  geom_hline(yintercept=0, 
             color = "darkgreen", size=1.5) +
  geom_bar(position=position_dodge(0.9), stat="identity",
           colour="black", # Use black outlines,
           size=.3) +      # Thinner lines
  scale_y_continuous(breaks = seq(-0.7, 0.7, by=0.1), limits = c(-0.7,0.7), labels=fmt_dcimals(1)) +
  scale_x_discrete(labels = c("Adversarial\nDebiasing", "Multilayer\nPerceptron", "Prejudice\nRemover","Ensemble\nClassifier")) +
  geom_point(data=eqopp_rew_long, position = position_nudge(x=0.22), size=3, alpha=0.6, shape=18, show.legend = FALSE) +
  geom_point(data=eqopp_nonrew_long, position = position_nudge(x=-0.22), size=3, alpha=0.6, shape=18, show.legend = FALSE) +
  theme(legend.position = c(0.79,0.12),
        axis.ticks.x = element_blank(),
        axis.title.x = element_text(margin = margin(t = 10, r = 0, b = 0, l = 0))) +
  labs(x="Algorithm", y="Equal Opportunity Difference")

############ Average Odds Difference ##################

avg_nonrew <- data.frame("advDebias" = advDebias$Average.Odds.Difference,
                           "neural" = neural$Average.Odds.Difference,
                           "prej" = prej$Average.Odds.Difference,
                           "ensemble" = ensemble$Average.Odds.Difference)
avg_rew <- data.frame("advDebias" = advDebiasRew$Average.Odds.Difference,
                        "neural" = neuralRew$Average.Odds.Difference,
                        "prej" = prejRew$Average.Odds.Difference,
                        "ensemble" = ensembleRew$Average.Odds.Difference)
avg_nonrew_long <- stack(avg_nonrew)
avg_nonrew_long['condition'] <- "non-reweighed"
avg_rew_long <- stack(avg_rew)
avg_rew_long['condition'] <- "reweighed"

avg <- rbind(avg_nonrew_long, avg_rew_long)

avg_summary <- summarySE(avg, measurevar = "values", groupvars = c("condition", "ind"))

avg_plot <- ggplot(avg_summary, aes(x=factor(ind), y=values, fill=condition)) +
  geom_rect(aes(xmin = -Inf, xmax = Inf, ymin = -0.1, ymax = 0.1),
            fill = "palegreen2", alpha = 0.2,color = NA) +
  geom_hline(yintercept=0, 
             color = "darkgreen", size=1.5) +
  geom_bar(position=position_dodge(0.9), stat="identity",
           colour="black", # Use black outlines,
           size=.3) +      # Thinner lines
  scale_y_continuous(breaks = seq(-0.6, 0.3, by=0.1), limits = c(-0.6,0.3), labels=fmt_dcimals(1)) +
  scale_x_discrete(labels = c("Adversarial\nDebiasing", "Multilayer\nPerceptron", "Prejudice\nRemover","Ensemble\nClassifier")) +
  geom_point(data=avg_rew_long, position = position_nudge(x=0.22), size=3, alpha=0.6, shape=18, show.legend = FALSE) +
  geom_point(data=avg_nonrew_long, position = position_nudge(x=-0.22), size=3, alpha=0.6, shape=18, show.legend = FALSE) +
  theme(legend.position = c(0.79,0.12),
        axis.ticks.x = element_blank(),
        axis.title.x = element_text(margin = margin(t = 10, r = 0, b = 0, l = 0))) +
  labs(x="Algorithm", y="Average Odds Difference")

########### Disparate Impact ##############

disp_nonrew <- data.frame("advDebias" = advDebias$Disparate.Impact,
                         "neural" = neural$Disparate.Impact,
                         "prej" = prej$Disparate.Impact,
                         "ensemble" = ensemble$Disparate.Impact)
disp_rew <- data.frame("advDebias" = advDebiasRew$Disparate.Impact,
                      "neural" = neuralRew$Disparate.Impact,
                      "prej" = prejRew$Disparate.Impact,
                      "ensemble" = ensembleRew$Disparate.Impact)
disp_nonrew_long <- stack(disp_nonrew)
disp_nonrew_long['condition'] <- "non-reweighed"
disp_rew_long <- stack(disp_rew)
disp_rew_long['condition'] <- "reweighed"

disp <- rbind(disp_nonrew_long, disp_rew_long)

disp_summary <- summarySE(disp, measurevar = "values", groupvars = c("condition", "ind"))

disp_plot <- ggplot(disp_summary, aes(x=factor(ind), y=values, fill=condition)) +
  geom_rect(aes(xmin = -Inf, xmax = Inf, ymin = 0.8, ymax = 1.2),
            fill = "palegreen2", alpha = 0.2,color = NA) +
  geom_hline(yintercept=1, 
             color = "darkgreen", size=1.5) +
  geom_bar(position=position_dodge(0.9), stat="identity",
           colour="black", # Use black outlines,
           size=.3) +      # Thinner lines
  scale_y_continuous(breaks = seq(0, 1.6, by=0.1), limits = c(0,1.6), labels=fmt_dcimals(1)) +
  scale_x_discrete(labels = c("Adversarial\nDebiasing", "Multilayer\nPerceptron", "Prejudice\nRemover","Ensemble\nClassifier")) +
  geom_point(data=disp_rew_long, position = position_nudge(x=0.22), size=3, alpha=0.6, shape=18, show.legend = FALSE) +
  geom_point(data=disp_nonrew_long, position = position_nudge(x=-0.22), size=3, alpha=0.6, shape=18, show.legend = FALSE) +
  theme(legend.position = c(0.79,0.87),
        axis.ticks.x = element_blank(),
        axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 10, r = 0, b = 0, l = 0))) +
  labs(x="Algorithm", y="Disparate Impact")

############ Theil Index ################

theil_nonrew <- data.frame("advDebias" = advDebias$Theil.Index,
                          "neural" = neural$Theil.Index,
                          "prej" = prej$Theil.Index,
                          "ensemble" = ensemble$Theil.Index)
theil_rew <- data.frame("advDebias" = advDebiasRew$Theil.Index,
                       "neural" = neuralRew$Theil.Index,
                       "prej" = prejRew$Theil.Index,
                       "ensemble" = ensembleRew$Theil.Index)
theil_nonrew_long <- stack(theil_nonrew)
theil_nonrew_long['condition'] <- "non-reweighed"
theil_rew_long <- stack(theil_rew)
theil_rew_long['condition'] <- "reweighed"

theil <- rbind(theil_nonrew_long, theil_rew_long)

theil_summary <- summarySE(theil, measurevar = "values", groupvars = c("condition", "ind"))

theil_plot <- ggplot(theil_summary, aes(x=factor(ind), y=values, fill=condition)) +
  geom_hline(yintercept=0, 
             color = "darkgreen", size=1.5) +
  geom_bar(position=position_dodge(0.9), stat="identity",
           colour="black", # Use black outlines,
           size=.3) +      # Thinner lines
  scale_y_continuous(breaks = seq(0, 0.4, by=0.05), limits = c(0,0.4), labels=fmt_dcimals(2)) +
  scale_x_discrete(labels = c("Adversarial\nDebiasing", "Multilayer\nPerceptron", "Prejudice\nRemover","Ensemble\nClassifier")) +
  geom_point(data=theil_rew_long, position = position_nudge(x=0.22), size=3, alpha=0.6, shape=18, show.legend = FALSE) +
  geom_point(data=theil_nonrew_long, position = position_nudge(x=-0.22), size=3, alpha=0.6, shape=18, show.legend = FALSE) +
  theme(legend.position = c(0.79,0.87),
        axis.ticks.x = element_blank(),
        axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 10, r = 0, b = 0, l = 0))) +
  labs(x="Algorithm", y="Theil Index")

########## Multiplot ##################

# include multiplot if grouping of graphs needed
# multiplot(accuracy_plot, mean_diff_plot, avg_plot, eqopp_plot, disp_plot, theil_plot, cols=3)
