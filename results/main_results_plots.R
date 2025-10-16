#Code with the updated plots
stop("Stop you have violated the law")
#Single monitor explorations
setwd("~/Library/CloudStorage/OneDrive-Personal/Coding/MARS/combing_monitors_private")

if (!require(tidyverse)) install.packages("tidyverse"); library(tidyverse)
if (!require(magrittr)) install.packages("magrittr"); library(magrittr)

library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)
library(skimr)
library(modelsummary)
library(fixest)
library(ks)
library(readr)
myTheme <- theme(plot.title = element_text(size = 14),
                 panel.background = element_rect(fill = '#F2F2ED'),
                 legend.text = element_text(size = 10),
                 plot.subtitle = element_text(size = 12),
                 axis.title = element_text(size = 12),
                 axis.text = element_text(size = 12, colour = 'black'),
                 legend.position = "right",
                 legend.background = element_rect(linetype = 3,size = 0.5, color = 'black', fill = 'grey94'),
                 legend.key = element_rect(size = 0.5, linetype = 1, color = 'black'))

#I also have some nice colors that I use in my various graphs.
nicepurp <- "#A88DBF"
niceblue <- '#38A5E0'
nicegreen <- '#A3DCC0'

custom_colors <- c("#2ECC71", "#A3E635", "#F4D03F", "#F39C12", "#E74C3C", "#C0392B", "#0072B2", "#CC79A7")


ggsaver <- function(filename, sc = 1) {
  ggsave(filename = paste0("results/graphs/", filename, ".png"), width = 6, height = 4, scale = sc)
}


plot_likelihood_ratio <- function(attack_scores, 
                                  honest_scores, 
                                  adjust_bw = 1,
                                  common_x = seq(0,10,0.1),
                                  line_color = "#C0392B", line_width = 1) {
  # Fit KDE models using ks package
  kde_attack <- ks::kde(x = attack_scores, h = adjust_bw * bw.nrd0(attack_scores))
  kde_honest <- ks::kde(x = honest_scores, h = adjust_bw * bw.nrd0(honest_scores))
  
  # Evaluate densities at the common points
  attack_y <- predict(kde_attack, x = matrix(common_x, ncol = 1))
  honest_y <- predict(kde_honest, x = matrix(common_x, ncol = 1))
  
  # Create data frame for likelihood ratio plot, adding small epsilon to avoid division by zero
  likelihood_ratio_df <- data.frame(
    x = common_x,
    ratio = attack_y / (honest_y + .Machine$double.eps) # Add epsilon for numerical stability
  )
  
  # Filter out extreme ratio values potentially caused by near-zero honest density
  likelihood_ratio_df <- likelihood_ratio_df[is.finite(likelihood_ratio_df$ratio), ]
  
  # Create the plot
  p <- ggplot(likelihood_ratio_df, aes(x = x, y = ratio)) +
    geom_line(color = line_color, linewidth = line_width) +
    myTheme
  return(p)
}




# Original histogram with added density curves
alpha_attack <- read_csv("data/processed_monitor_scores/BCB2_llama3170B_5_attack.csv")
alpha_honest <- read_csv("data/processed_monitor_scores/BCB_llama3170B_5_honest.csv")
alpha_oos_attack <- read_csv("data/processed_monitor_scores/APPS3_llama3170B_attack.csv")
alpha_oos_attack2 <- read_csv("data/processed_monitor_scores/APPS2_llama3170B_5_attack.csv")

alpha_all <- alpha_attack %>% mutate(Output = "Misaligned") %>%
  add_row(alpha_honest %>% mutate(Output = "Aligned"))

alpha_attacks <- alpha_attack %>% mutate(Output = "Training") %>%
  add_row(alpha_oos_attack %>% mutate(Output = "OOD hard")) %>%
  add_row(alpha_oos_attack2 %>% mutate(Output = "OOD well-behaved"))

# Plot 1: Histogram with density curves
ggplot(alpha_all, aes(x = score_fuzzed, fill = Output)) + 
  geom_histogram(aes(y = ..density..), binwidth = 0.5, boundary = 0.25, 
                 position = "identity",
                 color = 'black', alpha = 0.5) +
  myTheme +
  scale_fill_manual(values = c("#2ECC71","#C0392B")) + 
  scale_color_manual(values = c("#2ECC71","#C0392B")) +
  labs(title = "Distribution of Llama 3.1 70B monitor scores on training data",
       subtitle = "Notice the long right tail of aligned outputs",
       y = "Density", x = "Monitor score", 
       fill = "o1-mini output",
       color = "Output Type") +
  theme(legend.position = "bottom") + 
  scale_x_continuous(breaks = seq(0,9,1), limits = c(0,9))

ggsaver("onemon_hist", sc = 1)


######
#Experiment one


np_v_naive <- read_csv("results/np_vs_naive_no_resample.csv")

np_v_naive_mean_sums <- np_v_naive %>% filter(seed > 0) %>% group_by(Y, strategy) %>% summarize(
  samples = n(),
  insample_safety_mean = mean(insample_safety),
  oos_safety_mean = mean(oos_safety),
  iid_test_safety_mean = mean(iid_test_safety),
  in_sample_se = sd(insample_safety),
  oos_se = sd(oos_safety),
  oos_025_ptile = quantile(oos_safety, 0.025),
  oos_05_ptile = quantile(oos_safety, 0.05),
  oos_95_ptile = quantile(oos_safety, 0.95),
  oos_975_ptile = quantile(oos_safety, 0.975),
  iid_test_safety_se = sd(iid_test_safety),
  iid_test_safety_025_ptile = quantile(iid_test_safety, 0.025),
  iid_test_safety_05_ptile = quantile(iid_test_safety, 0.05),
  iid_test_safety_95_ptile = quantile(iid_test_safety, 0.95),
  iid_test_safety_975_ptile = quantile(iid_test_safety, 0.975),
  audit_fpr_mean = mean(audit_fpr),
  audit_budget_share_mean = mean(audit_budget_share),
  audit_fpr_se = sd(audit_fpr),
  audit_budget_share_se = sd(audit_budget_share),
  use_two_monitor_prob_mean = mean(use_two_monitor_prob),
  use_two_monitor_prob_se = sd(use_two_monitor_prob)
)


np_v_naive_mean_sums %>% mutate(FPR = (Y - 1)/20) %>% 
  ggplot(aes(x = FPR, y = iid_test_safety_mean, 
             color = strategy, shape = strategy)) +
  geom_point(size =3) + geom_line(linewidth = 0.7) + 
  geom_errorbar(aes(ymin = iid_test_safety_05_ptile,
                    ymax = iid_test_safety_95_ptile), width = 0.0015) +
  myTheme + 
  scale_x_reverse(breaks = seq(0,1,0.01), labels = scales::percent)+
  scale_y_continuous(breaks = seq(0,1,0.2), labels = scales::percent)+
  labs(title = "Neyman-Pearson policy achieves higher safety on i.i.d. test set",
       subtitle = "Naive single monitor audits the top X% of scores",
       caption = "Using Llama 3.1 70B as monitor. Kernel width calculated using Silverman's rule. 90% confidence intervals.",
       y = "Mean safety on i.i.d. test set", x = "Audit False Positive Rate", 
       color = "Policy class", shape = "Policy class") + 
  scale_color_manual(values = c("#0072B2", "#CC79A7")) + 
  theme(legend.position = 'bottom')
ggsaver('onemon_insample', sc = 1.12)

ggplot(alpha_attacks, aes(x = score_fuzzed)) + 
  geom_histogram(aes(y = ..density..), binwidth = 1, boundary = 0, 
                 position = "identity",
                 color = 'black', fill = niceblue) +
  myTheme + facet_wrap(~Output) + 
  labs(title = "Llama 3.1 70B monitor scores on attack distributions",
       subtitle = "OOD well behaved has less of a winner's curse dip compared to training", 
       y = "Density", x = "Monitor score") +
  theme(legend.position = "bottom",axis.title = element_text(size = 12),
        axis.text = element_text(size = 12, colour = 'black'),
        strip.text = element_text(size = 12)) + 
  scale_x_continuous(breaks = seq(0,10,2), limits = c(0,9))
ggsaver('onemon_triplehist', sc = 1.2)

np_v_naive_mean_sums%>%mutate(FPR = (Y - 1)/20) %>% 
  ggplot(aes(x = FPR, y = oos_safety_mean, 
             color = strategy, shape = strategy)) +
  geom_point(size = 3) + geom_line(linewidth = 0.7) +
  geom_errorbar(aes(ymin = oos_05_ptile,
                    ymax = oos_95_ptile), width = 0.002) +
  myTheme + 
  scale_x_reverse(breaks = seq(0,1,0.01), labels = scales::percent)+
  scale_y_continuous(breaks = seq(0,1,0.2), labels = scales::percent)+
  labs(title = "Naive single monitor does better on OOD well-behaved",
       subtitle = "OOD well-behaved has more mislaigned output around 8",
       caption = "Using Llama 3.1 70B as monitor. Kernel width calculated using Silverman's rule. 90% confidence intervals.",
       y = "Mean safety on OOD well-behaved", x = "Audit False Positive Rate", 
       color = "Policy class", shape = "Policy class") + 
  scale_color_manual(values = c("#0072B2", "#CC79A7")) + 
  theme(legend.position = 'bottom')
ggsaver('onemon_oos', sc = 1.12)

######
#With stocastick

nps_v1 <- read_csv("results/np_naive_stochastic.csv")

nps_v1_sums <- nps_v1 %>% filter(seed > 0) %>% group_by(Y, strategy) %>% summarize(
  samples = n(),
  insample_safety_mean = mean(insample_safety),
  oos_safety_mean = mean(oos_safety),
  iid_test_safety_mean = mean(iid_test_safety),
  oos_se = sd(oos_safety),
  iid_test_safety_se = sd(iid_test_safety),
  iid_test_safety_025_ptile = quantile(iid_test_safety, 0.025),
  iid_test_safety_05_ptile = quantile(iid_test_safety, 0.05),
  iid_test_safety_95_ptile = quantile(iid_test_safety, 0.95),
  iid_test_safety_975_ptile = quantile(iid_test_safety, 0.975),
  oos_025_ptile = quantile(oos_safety, 0.025),
  oos_05_ptile = quantile(oos_safety, 0.05),
  oos_95_ptile = quantile(oos_safety, 0.95),
  oos_975_ptile = quantile(oos_safety, 0.975)
)

nps_v1_sums%>%mutate(
  strategy = case_when(strategy == "Naive single monitor" ~ "Π_naive",
                       strategy == "Neyman-Pearson" ~ "Π_NP",
                       strategy == "Neyman-Pearson Stochastic" ~ "Π_NPS")) %>%
  ggplot(aes(x = Y, y = iid_test_safety_mean, 
             color = strategy, shape = strategy)) +
  geom_point(size = 3) + geom_line(linewidth = 0.7) + 
  geom_errorbar(aes(ymin = iid_test_safety_05_ptile,
                    ymax = iid_test_safety_95_ptile), width = 0.02) +
  myTheme + 
  scale_x_reverse(labels = scales::dollar)+
  scale_y_continuous(breaks = seq(0,1,0.2), labels = scales::percent)+
  labs(title = "Neyman Pearson stochastic achieves the highest safety",
       subtitle = "Audit costs 20x monitoring, which is normalized to cost $1",
       caption = "Kernel width calculated using Silverman's rule. 90% confidence intervals.",
       y = "Mean safety on i.i.d. test set", x = "Total budget", 
       color = "Policy class", shape = "Policy class") + 
  theme(legend.position = 'bottom', legend.text = element_text(size = 14), 
        legend.title = element_text(size = 14)) + 
  scale_color_manual(values = c("#0072B2", "#CC79A7", "#F39C12"))
ggsaver('nps_ca20', sc = 1.05)


nps_v1_sums%>%mutate(
  strategy = case_when(strategy == "Naive single monitor" ~ "Π_naive",
                       strategy == "Neyman-Pearson" ~ "Π_NP",
                       strategy == "Neyman-Pearson Stochastic" ~ "Π_NPS")) %>%
  ggplot(aes(x = Y, y = oos_safety_mean, 
             color = strategy, linetype = strategy)) +
  geom_point(size = 3) + geom_line(linewidth = 0.7) + 
  geom_errorbar(aes(ymin = oos_05_ptile,
                    ymax = oos_95_ptile), width = 0.02) +
  myTheme + 
  scale_x_reverse(labels = scales::dollar)+
  scale_y_continuous(breaks = seq(0,1,0.2), labels = scales::percent)+
  labs(title = "Neyman Pearson stochastic achieves the highest safety",
       subtitle = "Audit costs 20x monitoring, which is normalized to cost $1",
       caption = "Kernel width calculated using Silverman's rule. 90% confidence intervals.",
       y = "Mean safety on i.i.d. test set", x = "Total budget", 
       color = "Policy class", linetype = "Policy class") + 
  theme(legend.position = 'bottom', legend.text = element_text(size = 12)) + 
  scale_color_manual(values = c("#0072B2", "#CC79A7", "#F39C12"))


nps_v2 <- read_csv("results/np_naive_stochastic_casweep.csv")


nps_v2_sums <- nps_v2 %>% filter(seed > 0) %>% group_by(Y,c_a, strategy) %>% summarize(
  samples = n(),
  insample_safety_mean = mean(insample_safety),
  oos_safety_mean = mean(oos_safety),
  audit_fpr_mean = mean(audit_fpr),
  iid_test_safety_mean = mean(iid_test_safety),
  oos_se = sd(oos_safety),
  iid_test_safety_se = sd(iid_test_safety),
  iid_test_safety_025_ptile = quantile(iid_test_safety, 0.025),
  iid_test_safety_05_ptile = quantile(iid_test_safety, 0.05),
  iid_test_safety_95_ptile = quantile(iid_test_safety, 0.95),
  iid_test_safety_975_ptile = quantile(iid_test_safety, 0.975)
)


nps_v2_sums%>% mutate(
  strategy = case_when(strategy == "Naive single monitor" ~ "Π_naive",
                       strategy == "Neyman-Pearson" ~ "Π_NP",
                       strategy == "Neyman-Pearson Stochastic" ~ "Π_NPS")) %>%
  ggplot(aes(x = c_a, y = iid_test_safety_mean, 
             color = strategy, shape = strategy)) +
  geom_point(size = 3) + geom_line(linewidth = 0.7)+ 
  geom_errorbar(aes(ymin = iid_test_safety_05_ptile,
                    ymax = iid_test_safety_95_ptile), width = 0.1) +
  myTheme + 
  scale_x_continuous(trans = c("log10"))+
  scale_y_continuous(breaks = seq(0,1,0.2), labels = scales::percent)+
  labs(title = "Gains to the stochastic policy decreases as audit costs increase",
       subtitle = "Audit false positive rate held constent at 1% for deterministic policies",
       caption = "Kernel width calculated using Silverman's rule. 90% confidence intervals.",
       y = "Mean safety on i.i.d. test set", x = "Cost of audit/cost of monitor (log scale)", 
       color = "Policy class", shape = "Policy class") + 
  theme(legend.position = 'bottom', legend.text = element_text(size = 14), 
        legend.title = element_text(size = 14)) + 
  scale_color_manual(values = c("#0072B2", "#CC79A7", "#F39C12"))
ggsaver('nps_casweep', sc = 1.05)


#######
# two monitor shenanigans. 
beta_attack <- read_csv("data/processed_monitor_scores/BCB2_llama318B_5_attack.csv")
beta_honest <- read_csv("data/processed_monitor_scores/BCB_llama318B_5_honest.csv")
beta_ood_attack <- read_csv("data/processed_monitor_scores/APPS3_llama318B_attack.csv")



training_scores <- beta_attack %>% mutate(letype = "Misaligned") %>%
  add_row(beta_honest %>% mutate(letype = "Aligned"))

training_scores %<>% mutate(beta_score = score_fuzzed) %>%
  cbind((alpha_attack %>% select(score_fuzzed) %>%
           add_row(
             alpha_honest %>% select(score_fuzzed))) %>% 
          rename(alpha_score = score_fuzzed))

ggplot(training_scores, 
       aes(x = alpha_score, y = beta_score, color = letype)) +
  geom_point(position = position_jitter(0.05,0.05), alpha = 0.5) +
  labs(title = "Scatter plot of monitor scores",
       subtitle = "Random noise added in to make it easier to read (jittered)",
       x = "Llama 3.1 70B", y = "Llama 3.1 8B", color = NULL) + 
  scale_color_manual(values = c("#2ECC71","#C0392B")) +
  myTheme + theme(legend.position = 'bottom')
ggsaver("scatter")


tm <- read_csv("results/two_mon_v1.csv")


tm_sums <- tm %>% filter(seed > 0) %>% group_by(Y, strategy) %>% summarize(
  samples = n(),
  insample_safety_mean = mean(insample_safety),
  oos_safety_mean = mean(oos_safety),
  audit_fpr_mean = mean(audit_fpr),
  iid_test_safety_mean = mean(iid_test_safety),
  oos_se = sd(oos_safety),
  iid_test_safety_se = sd(iid_test_safety),
  iid_test_safety_025_ptile = quantile(iid_test_safety, 0.025),
  iid_test_safety_05_ptile = quantile(iid_test_safety, 0.05),
  iid_test_safety_95_ptile = quantile(iid_test_safety, 0.95),
  iid_test_safety_975_ptile = quantile(iid_test_safety, 0.975),
  oos_025_ptile = quantile(oos_safety, 0.025),
  oos_05_ptile = quantile(oos_safety, 0.05),
  oos_95_ptile = quantile(oos_safety, 0.95),
  oos_975_ptile = quantile(oos_safety, 0.975)
)

tm_sums%>% filter(iid_test_safety_mean > 0) %>%
  mutate(
    strategy = case_when(strategy == "Neyman-Pearson Alpha" ~ "Llama 70B monitor only",
                         strategy == "Neyman-Pearson Beta" ~ "Llama 8B monitor only",
                         strategy == "One second monitor region" ~ "Using both monitors",
                         TRUE ~ "boo")) %>% filter(strategy != "boo") %>%
  ggplot(aes(x = Y, y = iid_test_safety_mean, 
             color = strategy, shape = strategy)) +
  geom_point(size = 3) + geom_line(linewidth = 0.7) + 
  geom_errorbar(aes(ymin = iid_test_safety_05_ptile,
                    ymax = iid_test_safety_95_ptile), width = 0.06) +
  myTheme + 
  scale_x_reverse(labels = scales::dollar, breaks = seq(0,5,0.5))+
  scale_y_continuous(breaks = seq(0,1,0.2), labels = scales::percent)+
  labs(title = "Safety v. Cost pareto frontier for one/two monitors",
       subtitle = "Audit costs 20 times the Llama 70B monitor, which is normalized to cost $1",
       y = "Safety on i.i.d. test set", x = "Total budget", 
       caption = "Llama 8B costs $0.21, which is based on the real-world relative cost to 70B. SE from 99 repeated train test splits.",
       color = NULL, linetype = NULL, shape = NULL) + 
  theme(legend.position = 'bottom') +
  scale_color_manual(
    values = c("#F39C12", "#22cf22", "#490092"))
ggsave("results/graphs/twom_intro.png", width = 12, height = 6, scale = 0.64)


tm_sums%>% filter(iid_test_safety_mean > 0) %>%
  mutate(
    strategy = case_when(strategy == "Neyman-Pearson Alpha" ~ "Π_NPR 70B monitor",
                         strategy == "Neyman-Pearson Beta" ~ "Π_NPR 8B monitor",
                         strategy == "One second monitor region" ~ "Π_SR (Single region)",
                         TRUE ~ "Π_TM (Two monitor)")) %>%
  ggplot(aes(x = Y, y = iid_test_safety_mean, 
             color = strategy,fill = strategy, shape = strategy)) +
  geom_point(size = 3) + geom_line(linewidth = 0.7) + 
  geom_errorbar(aes(ymin = iid_test_safety_05_ptile,
                    ymax = iid_test_safety_95_ptile), width = 0.06, alpha = 0.7) +
  myTheme + 
  scale_x_reverse(labels = scales::dollar, breaks = seq(0,5,0.5))+
  scale_y_continuous(breaks = seq(0,1,0.2), labels = scales::percent)+
  labs(title = "Calling the second monitor on some optimized region achieves the highest safety",
       subtitle = "Audit costs 20x the expensive monitor, which is normalized to cost $1",
       caption = "Kernel width calculated using Silverman's rule. 90% confidence intervals. SE from 99 repeated train test splits.",
       y = "Mean safety on i.i.d. test set", x = "Total budget", 
       color = "Policy class", shape = "Policy class", fill = "Policy class") + 
  theme(legend.position = 'bottom') +
  scale_color_manual(
    values = c("#F39C12", "#22cf22", "#490092", "#009999")) + 
  scale_fill_manual(
    values = c("#F39C12", "#22cf22", "#490092", "#009999")) + 
  scale_shape_manual(values = c(21,24,22,23))
ggsave("results/graphs/twom_iid.png", width = 12, height = 6, scale = 0.73)


ggplot(alpha_all %>% add_row(alpha_oos_attack %>% mutate(Output = "OOD Misaligned")) %>% 
         mutate(Output = ifelse(Output == "Misaligned", "Train Misaligned", Output)), 
       aes(x = score_fuzzed, fill = Output)) + 
  geom_histogram(aes(y = ..density..), binwidth = 1, boundary = 0.5, 
                 position = "identity",
                 color = 'black') +
  myTheme + facet_wrap(~Output) + 
  labs(title = "Distribution of Llama 3.1 70B monitor scores",
       subtitle = "OOD misaligned output drawn from Wen et al. (2025)", 
       y = "Density", x = "Monitor score", 
       fill = "Model Output")+
  theme(legend.position = "bottom") + 
  scale_x_continuous(breaks = seq(0,10,2), limits = c(0,10)) + 
  scale_fill_manual(values = c("#2ECC71","#C6A10B", "#C0392B"))
ggsave("results/graphs/ood_hist.png", width = 6, height = 4, scale = 1.1)

all_scores_w_ood <- beta_attack %>% mutate(letype = "Train Misaligned") %>%
  add_row(beta_honest %>% mutate(letype = "Aligned")) %>% 
  add_row(beta_ood_attack %>% mutate(letype = "OOD Misaligned")) %>%
  rename(beta_score= score_fuzzed) %>% select(-score) %>% cbind(
    alpha_attack %>% mutate(letypea = "Train Misaligned") %>%
      add_row(alpha_honest %>% mutate(letypea = "Aligned")) %>% 
      add_row(alpha_oos_attack %>% mutate(letypea = "OOD Misaligned")) %>%
      rename(alpha_score = score_fuzzed) %>% select(-score)
  )

ggplot(all_scores_w_ood, aes(x = alpha_score, y = beta_score, color = letype)) +
  geom_point(position = position_jitter(0.05,0.05), alpha = 0.6) +
  labs(title = "Scatter plot of monitor scores including OOD scores",
       subtitle = "Random noise added in to make it easier to read (jittered)",
       x = "Llama 3.1 70B", y = "Llama 3.1 8B", color = NULL, shape = NULL) + 
  scale_color_manual(values = c("#2ECC71","#C6A10B","#C0392B")) +
  myTheme + theme(legend.position = 'bottom')
ggsaver("scatter_ood")


tm_sums%>% filter(iid_test_safety_mean > 0) %>%
  mutate(
    strategy = case_when(strategy == "Neyman-Pearson Alpha" ~ "Π_NPR 70B monitor",
                         strategy == "Neyman-Pearson Beta" ~ "Π_NPR 8B monitor",
                         strategy == "One second monitor region" ~ "Π_SR (Single region)",
                         TRUE ~ "Π_TM (Two monitor)")) %>%
  ggplot(aes(x = Y, y = oos_safety_mean, 
             color = strategy, linetype = strategy)) +
  geom_point(size = 3) + geom_line(linewidth = 0.7) + 
  geom_errorbar(aes(ymin = oos_05_ptile,
                    ymax = oos_95_ptile), width = 0.08) +
  myTheme + 
  scale_x_reverse(labels = scales::dollar, breaks = seq(0,4,0.5))+
  scale_y_continuous(breaks = seq(0,1,0.1), labels = scales::percent)+
  labs(title = "Neyman Pearson stochastic performs well out of distribution",
       subtitle = "Audit costs 20x monitoring, which is normalized to cost $1",
       caption = "Kernel width calculated using Silverman's rule. 90% confidence intervals. SE from 99 repeated train test splits.",
       y = "Mean safety on OOD test set", x = "Total budget", 
       color = "Policy class", linetype = "Policy class") +
  scale_color_manual(
    values = c("#F39C12", "#22cf22", "#490092", "#009999")) + 
  theme(legend.position = 'bottom')
ggsave("results/graphs/twom_ood.png", width = 12, height = 6, scale = 0.73)



correct_df_iid_test <- tm %>% filter(seed > 0) %>% select(Y, seed, iid_test_safety, strategy) %>% 
  pivot_wider(names_from = strategy, id_cols = c('Y', 'seed'), values_from = iid_test_safety) %>%
  mutate(SR_m_Alpha = `One second monitor region` - `Neyman-Pearson Alpha`, 
         SR_m_Beta = `One second monitor region` - `Neyman-Pearson Beta`,
         SR_m_TM = `One second monitor region` - `Two monitors always`)

correct_df_iid_test %>% group_by(Y) %>% summarize(
  bootstrap_draws = n(),
  SR_m_Alpha_mean = mean(SR_m_Alpha),
  SR_m_Alpha_q05 = quantile(SR_m_Alpha, 0.05),
  SR_m_Alpha_q95 = quantile(SR_m_Alpha, 0.95),
  SR_m_Beta_mean = mean(SR_m_Beta),
  SR_m_Beta_q05 = quantile(SR_m_Beta, 0.05),
  SR_m_Beta_q95 = quantile(SR_m_Beta, 0.95),
  SR_m_TM_mean = mean(SR_m_TM, na.rm = TRUE),
  SR_m_TM_q05 = quantile(SR_m_TM, 0.05, na.rm = TRUE),
  SR_m_TM_q95 = quantile(SR_m_TM, 0.95, na.rm = TRUE),
  mean_tm = mean(`Two monitors always`)) %>% ungroup() %>% 
  mutate(SR_m_TM_mean = ifelse(mean_tm > 0, SR_m_TM_mean,NA),
         SR_m_TM_q05 = ifelse(mean_tm > 0, SR_m_TM_q05,NA),
         SR_m_TM_q95 = ifelse(mean_tm > 0, SR_m_TM_q95,NA)) %>%
  ggplot() + 
  geom_point(aes(x = Y, y = SR_m_TM_mean, color = "Π_SR - Π_TM", shape = "Π_SR - Π_TM"), size = 2) + 
  geom_line(aes(x = Y, y = SR_m_TM_mean, color = "Π_SR - Π_TM")) +
  geom_errorbar(aes(x = Y, ymin = SR_m_TM_q05, 
                    ymax = SR_m_TM_q95, color = "Π_SR - Π_TM"), width = 0.08) +
  geom_point(aes(x = Y, y = SR_m_Alpha_mean, color = "Π_SR - Π_NPR 70B", shape = "Π_SR - Π_NPR 70B"), size = 2) + 
  geom_line(aes(x = Y, y = SR_m_Alpha_mean, color = "Π_SR - Π_NPR 70B")) + 
  geom_errorbar(aes(x = Y, ymin = SR_m_Alpha_q05, 
                    ymax = SR_m_Alpha_q95, color = "Π_SR - Π_NPR 70B"), width = 0.08) + 
  geom_point(aes(x = Y, y = SR_m_Beta_mean, color = "Π_SR - Π_NPR 8B", shape = "Π_SR - Π_NPR 8B"), size = 2) + 
  geom_line(aes(x = Y, y = SR_m_Beta_mean, color = "Π_SR - Π_NPR 8B")) + 
  geom_errorbar(aes(x = Y, ymin = SR_m_Beta_q05, 
                    ymax = SR_m_Beta_q95, color = "Π_SR - Π_NPR 8B"), width = 0.08) + 
  myTheme + 
  labs(title = "Percentage point improvement in safety in using the Π_SR policy",
       subtitle = "Relative to other baseline policies. Higher is better.", color = NULL, shape = NULL, 
       y = "Improvement in i.i.d. test set safety", x = "Budget", 
       caption = "90% confidence intervals from 99 repeated train test splits") + 
  scale_y_continuous(labels = scales::percent, breaks = seq(0,1,0.1))+
  scale_x_continuous(labels = scales::dollar, breaks = seq(0,10,0.5))+
  theme(legend.position = 'bottom') + 
  scale_color_manual( values = c("#F39C12", "#22cf22","#009999"))
ggsave("results/graphs/twom_comparison.png", width = 12, height = 6, scale = 0.73)

######
#OOD 

#np no resample??


np_v_naive <- read_csv("results/np_vs_naive_no_resample_hard.csv")

np_v_naive_mean_sums <- np_v_naive %>% filter(seed > 0) %>% group_by(Y, strategy) %>% summarize(
  samples = n(),
  insample_safety_mean = mean(insample_safety),
  oos_safety_mean = mean(oos_safety),
  iid_test_safety_mean = mean(iid_test_safety),
  in_sample_se = sd(insample_safety),
  oos_se = sd(oos_safety),
  oos_025_ptile = quantile(oos_safety, 0.025),
  oos_05_ptile = quantile(oos_safety, 0.05),
  oos_95_ptile = quantile(oos_safety, 0.95),
  oos_975_ptile = quantile(oos_safety, 0.975),
  iid_test_safety_se = sd(iid_test_safety),
  iid_test_safety_025_ptile = quantile(iid_test_safety, 0.025),
  iid_test_safety_05_ptile = quantile(iid_test_safety, 0.05),
  iid_test_safety_95_ptile = quantile(iid_test_safety, 0.95),
  iid_test_safety_975_ptile = quantile(iid_test_safety, 0.975),
  audit_fpr_mean = mean(audit_fpr),
  audit_budget_share_mean = mean(audit_budget_share),
  audit_fpr_se = sd(audit_fpr),
  audit_budget_share_se = sd(audit_budget_share),
  use_two_monitor_prob_mean = mean(use_two_monitor_prob),
  use_two_monitor_prob_se = sd(use_two_monitor_prob)
)


np_v_naive_mean_sums%>%mutate(FPR = (Y - 1)/20) %>% 
  ggplot(aes(x = FPR, y = oos_safety_mean, 
             color = strategy, shape = strategy)) +
  geom_point(size = 3) + geom_line(linewidth = 0.7) +
  geom_errorbar(aes(ymin = oos_05_ptile,
                    ymax = oos_95_ptile), width = 0.002) +
  myTheme + 
  scale_x_reverse(breaks = seq(0,1,0.01), labels = scales::percent)+
  scale_y_continuous(breaks = seq(0,1,0.2), labels = scales::percent)+
  labs(title = "Naive single monitor does better on OOD well-behaved",
       subtitle = "OOD well-behaved has more mislaigned output around 8",
       caption = "Using Llama 3.1 70B as monitor. Kernel width calculated using Silverman's rule. 90% confidence intervals.",
       y = "Mean safety on OOD well-behaved", x = "Audit False Positive Rate", 
       color = "Policy class", shape = "Policy class") + 
  scale_color_manual(values = c("#0072B2", "#CC79A7")) + 
  theme(legend.position = 'bottom')

#np w stochastic


nps_v1 <- read_csv("results/np_naive_stochastic_hard.csv")

nps_v1_sums <- nps_v1 %>% filter(seed > 0) %>% group_by(Y, strategy) %>% summarize(
  samples = n(),
  insample_safety_mean = mean(insample_safety),
  oos_safety_mean = mean(oos_safety),
  iid_test_safety_mean = mean(iid_test_safety),
  oos_se = sd(oos_safety),
  iid_test_safety_se = sd(iid_test_safety),
  iid_test_safety_025_ptile = quantile(iid_test_safety, 0.025),
  iid_test_safety_05_ptile = quantile(iid_test_safety, 0.05),
  iid_test_safety_95_ptile = quantile(iid_test_safety, 0.95),
  iid_test_safety_975_ptile = quantile(iid_test_safety, 0.975),
  oos_025_ptile = quantile(oos_safety, 0.025),
  oos_05_ptile = quantile(oos_safety, 0.05),
  oos_95_ptile = quantile(oos_safety, 0.95),
  oos_975_ptile = quantile(oos_safety, 0.975)
)

nps_v1_sums%>%mutate(
  strategy = case_when(strategy == "Naive single monitor" ~ "Π_naive",
                       strategy == "Neyman-Pearson" ~ "Π_NP",
                       strategy == "Neyman-Pearson Stochastic" ~ "Π_NPS")) %>%
  ggplot(aes(x = Y, y = oos_safety_mean, 
             color = strategy, shape = strategy)) +
  geom_point(size = 3) + geom_line(linewidth = 0.7) + 
  geom_errorbar(aes(ymin = oos_05_ptile,
                    ymax = oos_95_ptile), width = 0.02) +
  myTheme + 
  scale_x_reverse(labels = scales::dollar)+
  scale_y_continuous(breaks = seq(0,1,0.2), labels = scales::percent)+
  labs(title = "Neyman Pearson stochastic achieves the highest safety",
       subtitle = "Audit costs 20x monitoring, which is normalized to cost $1",
       caption = "Kernel width calculated using Silverman's rule. 90% confidence intervals.",
       y = "Mean safety on i.i.d. test set", x = "Total budget", 
       color = "Policy class", shape = "Policy class") + 
  theme(legend.position = 'bottom', legend.text = element_text(size = 14), 
        legend.title = element_text(size = 14)) + 
  scale_color_manual(values = c("#0072B2", "#CC79A7", "#F39C12"))
ggsaver('nps_ca20', sc = 1.05)


#stochastic ca sweep
nps_v2 <- read_csv("results/np_naive_stochastic_casweep_hard.csv")


nps_v2_sums_oos <- nps_v2 %>% filter(seed > 0) %>% group_by(Y,c_a, strategy) %>% summarize(
  samples = n(),
  insample_safety_mean = mean(insample_safety),
  oos_safety_mean = mean(oos_safety),
  audit_fpr_mean = mean(audit_fpr),
  iid_test_safety_mean = mean(iid_test_safety),
  oos_se = sd(oos_safety),
  iid_test_safety_se = sd(iid_test_safety),
  oos_test_safety_025_ptile = quantile(oos_safety, 0.025),
  oos_test_safety_05_ptile = quantile(oos_safety, 0.05),
  oos_test_safety_95_ptile = quantile(oos_safety, 0.95),
  oos_test_safety_975_ptile = quantile(oos_safety, 0.975)
)


nps_v2_sums_oos%>% mutate(
  strategy = case_when(strategy == "Naive single monitor" ~ "Π_naive",
                       strategy == "Neyman-Pearson" ~ "Π_NP",
                       strategy == "Neyman-Pearson Stochastic" ~ "Π_NPS")) %>%
  ggplot(aes(x = c_a, y = oos_safety_mean, 
             color = strategy, shape = strategy)) +
  geom_point(size = 3) + geom_line(linewidth = 0.7)+ 
  geom_errorbar(aes(ymin = oos_test_safety_05_ptile,
                    ymax = oos_test_safety_95_ptile), width = 0.1) +
  myTheme + 
  scale_x_continuous(trans = c("log10"))+
  scale_y_continuous(breaks = seq(0,1,0.05), labels = scales::percent)+
  labs(title = "Gains to the stochastic policy decreases as audit costs increase",
       subtitle = "Audit false positive rate held constent at 1% for deterministic policies",
       caption = "Kernel width calculated using Silverman's rule. 90% confidence intervals.",
       y = "Mean safety on i.i.d. test set", x = "Cost of audit/cost of monitor (log scale)", 
       color = "Policy class", shape = "Policy class") + 
  theme(legend.position = 'bottom', legend.text = element_text(size = 14), 
        legend.title = element_text(size = 14)) + 
  scale_color_manual(values = c("#0072B2", "#CC79A7", "#F39C12"))
ggsaver('nps_casweep_oos', sc = 1.05)

