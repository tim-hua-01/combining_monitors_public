#Code for combining probe and black box monitor results, intended for alignment forum post
stop('stop in case i accidentally run the whole script')
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


beta_honest <- read_csv('data/processed_monitor_scores/mean_pairs_honest.csv')
beta_attack <- read_csv('data/processed_monitor_scores/mean_pairs_attack.csv')
alpha_honest <- read_csv('data/processed_monitor_scores/sonnet_honest.csv')
alpha_attack <- read_csv('data/processed_monitor_scores/sonnet_attack.csv')



alpha_all <- alpha_attack %>% mutate(Output = "Misaligned") %>%
  add_row(alpha_honest %>% mutate(Output = "Aligned"))

# Plot 1: Histogram with density curves
ggplot(alpha_all, aes(x = score_fuzzed, fill = Output)) + 
  geom_histogram(aes(y = ..density..), binwidth = 0.5, boundary = 0.25, 
                 position = "identity",
                 color = 'black', alpha = 0.5) +
  myTheme +
  scale_fill_manual(values = c("#2ECC71","#C0392B")) + 
  scale_color_manual(values = c("#2ECC71","#C0392B")) +
  labs(title = "Distribution of LLM monitor scores on training data",
       subtitle = "All data from Parrack et al. (2025)",
       y = "Density", x = "Monitor score", 
       fill = "Untrusted AI's output",
       color = "Output Type") +
  theme(legend.position = "bottom")



beta_scores <- beta_attack %>% mutate(letype = "Misaligned") %>%
  add_row(beta_honest %>% mutate(letype = "Aligned"))

ggplot(beta_scores, aes(x = score_fuzzed, fill = letype)) + 
  geom_histogram(aes(y = ..density..), binwidth = 0.5, boundary = 0.25, 
                 position = "identity",
                 color = 'black', alpha = 0.5) +
  myTheme +
  scale_fill_manual(values = c("#2ECC71","#C0392B")) + 
  scale_color_manual(values = c("#2ECC71","#C0392B")) +
  labs(title = "Distribution of probe monitor scores on training data",
       subtitle = "All data from Parrack et al. (2025)",
       y = "Density", x = "Monitor score", 
       fill = "Untrusted AI's output",
       color = "Output Type") +
  theme(legend.position = "bottom")

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
  labs(title = "Scatter plot of monitor scores on training data",
       subtitle = "Random noise added in to make it easier to read (jittered)",
       x = "Black box monitor", y = "Probe monitor", color = NULL) + 
  scale_color_manual(values = c("#2ECC71","#C0392B")) +
  myTheme + theme(legend.position = 'bottom')


comb_graph <- read_csv('experiments/probe_comb_exp.csv') %>% 
  add_row(read_csv('experiments/probe_comb_exp2.csv')) %>%
  add_row(read_csv('experiments/probe_comb_exp3.csv'))
comb_graph %>% ggplot(aes(x = Y, y = iid_test_safety, 
                          color = strategy, shape = strategy)) +
  geom_point(size = 3) + geom_line(linewidth = 0.7) +
  myTheme + 
  scale_x_reverse(labels = scales::dollar, breaks = seq(0,5,0.5))+
  scale_y_continuous(breaks = seq(0,1,0.2), labels = scales::percent)+
  labs(title = "Optimal combination of probe and black box monitor is Pareto Optimal",
       subtitle = "Safety v. cost for probe-only, black box-oonly, and combined monitoring protocol",
       y = "Safety (i.e., recall over misaligned)", x = "Total budget", 
       caption = "Using Π_NPR single monitor strategy (see paper)",
       color = NULL, linetype = NULL, shape = NULL) + 
  theme(legend.position = 'bottom') +
  scale_color_manual(
    values = c("#F39C12", "#22cf22", "#490092"))



