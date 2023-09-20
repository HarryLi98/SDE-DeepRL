library(tidyverse)
library(latex2exp)

file_path <- "C:/Users/Harry/OneDrive - Imperial College London/Imperial/Project/Python/SDETools.py/charts"
description1 <- "N=9,a=1,sig=0"
description2 <- "N=9,a=1,sig=0.1"
data_path1 <- file.path(file_path, paste0(description1, "-tag-epoch_loss.csv"))
data_path2 <- file.path(file_path, paste0(description2, "-tag-10k-epoch_loss.csv"))

data1 <- read.csv(data_path1)
data2 <- read.csv(data_path2)

data1 |> 
  ggplot(aes(x = epoch, y = log10(value))) +
  geom_line() + 
  ylab("log10 loss") +
  labs(title = TeX("Loss function with $\\sigma = 0$")) + 
  theme_classic() +
  theme(text = element_text(size = 20, color = "black"),
        axis.text = element_text(size = 20, color = "black"),
        plot.title = element_text(hjust = 0.5))

ggsave(file.path(file_path, paste0(description1, "-loss.png")), 
       width = 8, height = 5, dpi = 300)

data2 |>
  ggplot(aes(x = Step, y = log10(Smoothed.value))) +
  geom_line() + 
  xlab("epoch") +
  ylab("log10 loss") +
  labs(title = TeX("Loss function with $\\sigma = 0.1$")) + 
  theme_classic() +
  theme(text = element_text(size = 20, color = "black"),
        axis.text = element_text(size = 20, color = "black"),
        plot.title = element_text(hjust = 0.5))

ggsave(file.path(file_path, paste0(description2, "-loss.png")), 
       width = 8, height = 5, dpi = 300)
