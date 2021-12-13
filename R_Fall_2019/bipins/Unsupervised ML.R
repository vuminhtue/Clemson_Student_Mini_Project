library(ggplot2)
library(factoextra)
library(purrr)

df = read.csv("aniso.csv")

head(df)


########################
# Check the data
########################

ggplot(df, aes(x=V1, y=V2)) + geom_point()


#######################
# K-Means Clustering
#######################

set.seed(1234)
km = kmeans(df,3,nstart=20)
table(km$cluster,df$V2)
fviz_cluster(km, data = df)  # view as a cluster plot

set.seed(1234)
db = fpc::dbscan(df, eps = 0.15, MinPts = 5)