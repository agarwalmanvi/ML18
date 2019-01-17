setwd("/home/manvi/Documents/ML18-ensemble")

# add these lines to your python code
# df_compas_reweigh.to_csv("df_compas_reweigh.csv", encoding='utf-8')
# df_compas_nonreweigh.to_csv("df_compas_nonreweigh.csv", encoding='utf-8')

rew <- read.csv("df_compas_reweigh.csv")
non_rew <- read.csv("df_compas_nonreweigh.csv")

# remove eq_opp_diff and avg_odds_diff for now
rew <- rew[-c(4,5), ]
non_rew <- non_rew[-c(4,5), ]

status <- c("Non-Reweigh","Non-Reweigh","Non-Reweigh","Non-Reweigh")
non_rew$Status <- status
status <- c("Reweigh","Reweigh","Reweigh","Reweigh")
rew$Status <- status

accuracy <- rbind(non_rew[c(1),], rew[c(1),])
accuracy_df <- data.frame(t(accuracy[-6]))
colnames(accuracy_df) <- accuracy[, 6]
accuracy_df = accuracy_df[-1,]
accuracy_df$algo <- c(1,2,3,4)
accuracy_long <- melt(accuracy_df,
                 id.vars = "algo",
                 measure.vars = c("Non-Reweigh","Reweigh"),
                 variable.name = "condition")
accuracy_long$value <- as.numeric(as.character(accuracy_long$value)) * 100
accuracy_long$metric <- 1
ggplot(data=accuracy_long, aes(x=factor(algo), y=value, group=variable, colour=variable)) +
  geom_line() +
  geom_point() +
  labs (x = "Algorithm",
        y = "Accuracy") +
  scale_x_discrete(labels = c("1" = "Adversarial\nDebiasing", "2" = "Prejudice\nRemover", "3" = "Neural\nNetwork", "4" = "Ensemble\nClassifier"))

mean_diff <- rbind(non_rew[c(2),], rew[c(2),])
mean_diff_df <- data.frame(t(mean_diff[-6]))
colnames(mean_diff_df) <- mean_diff[, 6]
mean_diff_df = mean_diff_df[-1,]
mean_diff_df$algo <- c(1,2,3,4)
mean_diff_long <- melt(mean_diff_df,
                      id.vars = "algo",
                      measure.vars = c("Non-Reweigh","Reweigh"),
                      variable.name = "condition")
mean_diff_long$metric <- 2
mean_diff_long$value <- as.numeric(as.character(mean_diff_long$value))
ggplot(data=mean_diff_long, aes(x=factor(algo), y=value, group=variable, colour=variable), axes = FALSE) +
  geom_line() +
  geom_point() +
  geom_rect(aes(xmin = -Inf, xmax = Inf, ymin = -0.1, ymax = 0.1),
            fill = "palegreen2", alpha = 0.05,color = NA) +
  geom_hline(yintercept=0, linetype="dashed", 
             color = "palegreen3", size=1.5) +
  labs (x = "Algorithm",
        y = "Mean Difference") +
  scale_x_discrete(labels = c("1" = "Adversarial\nDebiasing", "2" = "Prejudice\nRemover", "3" = "Neural\nNetwork", "4" = "Ensemble\nClassifier"))


disp_impact <- rbind(non_rew[c(3),], rew[c(3),])
disp_impact_df <- data.frame(t(disp_impact[-6]))
colnames(disp_impact_df) <- disp_impact[, 6]
disp_impact_df = disp_impact_df[-1,]
disp_impact_df$algo <- c(1,2,3,4)
disp_impact_long <- melt(disp_impact_df,
                      id.vars = "algo",
                      measure.vars = c("Non-Reweigh","Reweigh"),
                      variable.name = "condition")
disp_impact_long$value <- as.numeric(as.character(disp_impact_long$value))
disp_impact_long$metric <- 3
ggplot(data=disp_impact_long, aes(x=factor(algo), y=value, group=variable, colour=variable)) +
  geom_line() +
  geom_point() +
  geom_rect(aes(xmin = -Inf, xmax = Inf, ymin = 0.8, ymax = 1.2),
            fill = "palegreen2", alpha = 0.05,color = NA) +
  geom_hline(yintercept=1, linetype="dashed", 
             color = "palegreen3", size=1.5) +
  labs (x = "Algorithm",
        y = "Disparate Impact") +
  scale_x_discrete(labels = c("1" = "Adversarial\nDebiasing", "2" = "Prejudice\nRemover", "3" = "Neural\nNetwork", "4" = "Ensemble\nClassifier"))


theil <- rbind(non_rew[c(4),], rew[c(4),])
theil_df <- data.frame(t(theil[-6]))
colnames(theil_df) <- theil[, 6]
theil_df = theil_df[-1,]
theil_df$algo <- c(1,2,3,4)
theil_long <- melt(theil_df,
                      id.vars = "algo",
                      measure.vars = c("Non-Reweigh","Reweigh"),
                      variable.name = "condition")
theil_long$metric <- 4
theil_long$value <- as.numeric(as.character(theil_long$value))
ggplot(data=theil_long, aes(x=factor(algo), y=value, group=variable, colour=variable), axes = FALSE) +
  geom_line() +
  geom_point() +
  geom_hline(yintercept=0, linetype="dashed", 
             color = "palegreen3", size=1.5) +
  labs (x = "Algorithm",
        y = "Theil Index") +
  scale_x_discrete(labels = c("1" = "Adversarial\nDebiasing", "2" = "Prejudice\nRemover", "3" = "Neural\nNetwork", "4" = "Ensemble\nClassifier"))

