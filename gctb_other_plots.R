sample = 'L98'
cellSNP.base = read.table(gzfile (str_c("~/data/maester/GCTB/", sample, "/maester_cellSNP/cellSNP.base.vcf.gz")))
colnames(cellSNP.base) = c("chrom", "pos", "id", "ref", "alt", "qual", "filter", "info")
cellSNP.base = cellSNP.base %>% 
  separate(info, c("ad", "dp", "oth"), sep = ";") %>% 
  mutate(ad = str_remove(ad, "AD="),
         dp = str_remove(dp, "DP="),
         oth = str_remove(oth, "OTH="),
         ad = as.numeric(ad),
         dp = as.numeric(dp),
         oth = as.numeric(oth))

cellSNP.base.scRNA = read.table(gzfile(str_c("/home/linxy29/data/maester/GCTB/", sample, "/scRNA_cellSNP/cellSNP.base.vcf.gz")))
colnames(cellSNP.base.scRNA) = c("chrom", "pos", "id", "ref", "alt", "qual", "filter", "info")
cellSNP.base.scRNA = cellSNP.base.scRNA %>% 
  separate(info, c("ad", "dp", "oth"), sep = ";") %>% 
  mutate(ad = str_remove(ad, "AD="),
         dp = str_remove(dp, "DP="),
         oth = str_remove(oth, "OTH="),
         ad = as.numeric(ad),
         dp = as.numeric(dp),
         oth = as.numeric(oth))

cellSNP.base %>% 
  rename(MAESTER = dp) %>% 
  mutate(scRNA = cellSNP.base.scRNA$dp) %>% 
  ggplot(aes(x = pos)) +
  geom_col(aes(y = MAESTER, fill = "MAESTER")) +
  geom_col(aes(y = scRNA, fill = "scRNA")) +
  xlab("Position at the chrM") +
  ylab("Reads count") +
  ggtitle(str_c("The coverage of day ", sample, " sample"))
#ggsave(str_c("~/data/maester/GCTB/", sample, "/scRNA_MAESTER_coverage.png"),height = 8, width = 12)
ggsave(str_c("./plot/d", sample, "_scRNA_MAESTER_coverage.png"),height = 8, width = 12)