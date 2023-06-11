## 팝콘 - 청소년 팝업쉼터
# install.packages("readr")
# install.packages("readxl")
# install.packages("ggplot2")
library(readr)
library(readxl)
library(ggplot2)


# install.packages("dplyr")
library(dplyr)
## 청소년 상담내용
sd <- read.csv("여성가족부_청소년상담_내용.csv", header=T)
colnames(sd) <- c("상담내용","중학생","고등학생")

# 상담 수 합치는 파생변수 생성
sd$계 <- c(sd$중학생 + sd$고등학생)

# 14행이 총계라서 제거
sd <- sd[-14,]

sd %>% ggplot(aes(x=reorder(상담내용,계),
                  y=계, fill = 상담내용)) +
  ggtitle('청소년 상담내용') +
  xlab('상담내용') + ylab('인원') +
  geom_bar(stat="identity") + coord_flip()


## 2019~2020년 청소년 검색 관심키워드 워드클라우드
like <- read.csv("청소년 관심키워드.csv", header=T)

# 검색어와 조회수만 가져오기
like <- like[, c(4,5)]
colnames(like) <- c("관심키워드", "조회수")

# 동일 검색어 조회수 합계 구하기
li_agg <- aggregate(like$조회수,
                    by=list(합계=like$관심키워드),
                    FUN=sum)

# install.packages("wordcloud2")
library(wordcloud2)
wordcloud2(li_agg)


# 위와 동일
## 2019~2020년 청소년 검색 고민키워드 워드클라우드
wor <- read.csv("청소년 고민 순위.csv", header=T)
wor <- wor[, c(4, 5)]
colnames(wor) <- c("고민키워드", "조회수")
wor_agg <- aggregate(wor$조회수,
                    by=list(합계=wor$고민키워드),
                    FUN=sum)
wor_agg2 <- wor_agg[which(wor_agg$합계 != '친구'),]
wordcloud2(wor_agg2, size = 1.6)

wordcloud2(wor_agg2, size=1.6, color='random-light',
           backgroundColor="black")


## 친구라는 단어가 너무 독보적으로 조회수가 많아서 제거
# 일단 조회수 내림차순으로 정렬하기
wor_agg <- wor_agg[order(wor_agg$x, decreasing = T),]
wor_agg <- wor_agg[-1,]
wordcloud2(wor_agg)


## 고민내용 감정분석하기
# 군산대학교 감정사전
dic <- read_csv("knu_sentiment_lexicon.csv")

# 고민키워드 - 감정사전 점수 부여 (다른 컬럼명일 때 "=" 사용)
word_comment <- wor %>% 
  left_join(dic, by= c("고민키워드" = "word")) %>% 
  mutate(polarity=ifelse(is.na(polarity), 0, polarity))

# 긍정, 부정 표시 (mutate 사용 - 변수추가)
word_comment <- word_comment %>% 
  mutate(sentiment=ifelse(polarity >= 1,"pos", 
                          ifelse(polarity <= -1, "neg", "neu")))

# 긍정, 부정, 중립 개수 알아보기 
word_comment %>% count(sentiment)

# 긍정, 부정단어 추출
top10_sentiment <- word_comment %>% 
  filter(sentiment !="neu") %>% 
  count(sentiment, 고민키워드) %>%  
  group_by(sentiment) %>% 
  slice_max(n, n=10)

top10_sentiment
# 부정단어 20개, 긍정단어 5개

# 막대그래프
library(ggplot2)
ggplot(top10_sentiment, aes(x=reorder(고민키워드, n), y=n, fill=sentiment))+
  geom_col()+
  coord_flip()+
  geom_text(aes(label=n), hjust = -0.3)+
  facet_wrap(~sentiment, scales = "free")+
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.15)))+
  theme(axis.text.y = element_text(hjust = 1, size = 20)) +
  labs(x=NULL)  # x축 이름 삭제

# geom_text(aes(label=n)) - 그래프에 수치 적어주기

# facet_wrap(~sentiment, scales = "free") 
#       - 감정별 그래프 나눠주기(긍정, 부정)

# scale_y_continuous(expand = expansion(mult = c(0.05, 0.15)))
         # 막대 끝과 경계의 간격을 넓어지게(숫자 보이게)


## 부정 단어만 워드클라우드
neg <- top10_sentiment %>% filter(sentiment == "neg")
neg <- neg[,-1]
wordcloud2(neg)


## 구별 청소년 수
pop <- read_excel("청소년수_서울시.xlsx")
pop %>% ggplot(aes(x=reorder(지역, 학령인구),
                   y=학령인구, fill = 학령인구)) +
  ggtitle('지역구별 청소년 수') +
  xlab('지역') + ylab('청소년 수') +
  geom_bar(stat="identity")+coord_flip() +
  scale_fill_gradient(low='lightyellow', high='darkorange') +
  theme(axis.text.y = element_text(angle = 20, hjust = 1))
# scale_fill_gradient 추가, low와 high를 지정해서 색을 입힘
# y_label 20도 회전


## 청소년 수와, 청소년들이 많이 다닐 것 같은 곳 상관관계
## (PC방, 코인노래방, 독서실, 학원) 데이터에서 행정구 컬럼만들기

# 1. 학원
# 학원 데이터에 공백이 있음 - 공백은 모두 NA로 바꾸기
             # na.strings=c("","NA")
ac <- read.csv("학원.csv", header=T, na.strings=c("","NA"))
sum(is.na(ac$행정구역명))

# 결측값 제거
ac <- ac[complete.cases(ac),] 

# 행정구 빈도수 구하고 데이터프레임 변환
addr <- table(ac$행정구역명)
addr <- data.frame(addr)

# 컬럼명
colnames(addr) <- c("지역","학원")


# 위와 동일한 과정으로
# 2. 스터디카페
stu <- read.csv("스터디.csv", header=T)
str(stu)
sum(is.na(stu$행정구))

addr2 <- table(stu$행정구)
addr2 <- data.frame(addr2)

colnames(addr2) <- c("지역","독서실")


# 3. 코인노래방
coin <- read.csv("코인.csv")
head(coin)
sum(is.na(coin))

# 엑셀에서 이미 구 다음 두 개의 공백을 줌
addr3 <- substr(coin$도로명전체주소,7,10)   # 구추출
head(addr3)

# 공백 제거
addr_trim <- gsub(" ","",addr3)  

addr3 <- addr_trim %>% table() %>% data.frame()
head(addr3)
colnames(addr3) <- c('지역', '노래방수')


# 4. PC방
pc <- read.csv("pc방.csv", header=T, na.strings=c("","NA"))
str(pc)
sum(is.na(pc$주소))

pc <- pc[complete.cases(pc),]
str(pc)

addr4 <- substr(pc$주소, 7, 10)

addr_trim2 <- gsub(" ","",addr4)
head(addr_trim2)

addr4 <- addr_trim2 %>% table() %>% data.frame()
head(addr4)
colnames(addr4) <- c("지역","PC방수")


# 청소년 수와 PC방, 코인노래방, 독서실, 학원 합치기
   # pop, pc(addr4), coin(addr3), stu(addr2), ac(addr)
a <- merge(pop, addr, by = c("지역"))
b <- merge(a, addr2, by = c('지역'))

# by 생략가능?
c <- merge(b, addr3)
d <- merge(c, addr4)


# 상관관계구하기
point <- as.numeric(factor(d$지역))
pairs(d[,-1], col=point, pch = 19)
cor(d[,-1])

# 다중선형관계
fit <- lm(d[,-1],data=d)
summary(fit)

## 상관관계 P값 확인 방법
# 한번에 확인
install.packages("Hmisc")
library(Hmisc)
rcorr(as.matrix(d[,-1]), type = "pearson")

# 하나씩 확인
cor.test(d$학령인구, d$학원, method = 'pearson')
cor.test(d$학령인구, d$독서실, method = 'pearson')
cor.test(d$학령인구, d$노래방수, method = 'pearson')
cor.test(d$학령인구, d$PC방수, method = 'pearson')

## 학원(0.87), 독서실(0.84), 노래방은(0.61) 학생수와 상당한 상관관계가 있음을 볼 수 있다.


## 거리계산(구로구 학원, 독서실, 공원)

# install.packages("ggmap")
library(ggmap)

# 학원 위도, 경도 가져오기
# 데이터가 많아 시간이 걸려서 간단하게 밑에 사용
edu_add <- read_excel("구로학원.xlsx")

# 위도, 경도 가져와서 데이터만들기
register_google(key='AIzaSyArPg0YNhicPk6blDmoUcwGdBuafNPan6o')
# edu <- read_excel("구로구학원.xlsx")
# edu_add_code <- geocode(enc2utf8(edu$도로명주소))
# edu_add <- data.frame(edu,edu_add_code)


# 독서실 데이터(위도, 경도 포함)
study_add <- read_excel("구로독서실.xlsx")

# 독서실 위도, 경도 가져오기
# study <- read.csv("독서실_구로.csv", header=T)
# study_add_code <- geocode(enc2utf8(study$도로명주소))
# study_add <- data.frame(study,study_add_code)


# 공원데이터
park <- read_excel("녹지대.xlsx")
cen1 <- c(mean(park$lon), mean(park$lat))
park_map <- get_googlemap(center = cen1, 
                          maptype = "roadmap", 
                          zoom = 13)
ggmap(park_map) +
  geom_point(data = park, aes(x = lon, y = lat),
             colour = 'red', size = 4) +
  geom_text(data = park, aes(label = 공원명, vjust = -1, size = 13)) +
  geom_point(data = edu_add, aes(x = lon, y = lat),
             colour = "blue",size = 2) +
  geom_point(data = study_add, aes(x = lon, y = lat),
             colour = "purple", size = 2)

## 독서실하고 학원을 거리가 겹치는 게 많아 학원만 가지고 거리계산.


## 거리계산 패키지
install.packages("geosphere")
library(geosphere)

# 거리계산
mat <- distm(edu_add[,c('lon', 'lat')], park[,c('lon', 'lat')], fun = distVincentyEllipsoid)

# max.col(-mat) - which.min(mat)
edu_add$locality <- park$공원명[max.col(-mat)]

# 최소거리 추출 / apply 반복
edu_add$near_dist <- apply(mat, 1, min)
edu_add

a <- data.frame(table(edu_add$locality))
colnames(a) <- c("공원명", "학원수")
a <- t(a)

# k-means 사용 공원 개수, 위치 타당성 보기
# 군집 개수 3개
data <- edu_add[,c('lon', 'lat')]    # 비지도 - 정답빼고 데이터 형성
fit <- kmeans(x=data, centers = 3)    # 군집개수 3개
fit

fit$cluster         # 군집번호
fit$centers         # 군집 중심점 좌표

install.packages("cluster")
library(cluster)
# 군집 시각화
clusplot(data, fit$cluster, color = TRUE, 
         shade = TRUE, labels = 2, lines = 0)


# 이상값 제거 
data1 <- data[-c(98,284),]  

fit <- kmeans(x=data1, centers = 3)    # 군집개수 3개
fit

clusplot(data1, fit$cluster, color = TRUE, 
         shade = TRUE, labels = 2, lines = 0)


# 지도에 찍기
p1 <- subset(data1, fit$cluster==1)
p2 <- subset(data1, fit$cluster==2)
p3 <- subset(data1, fit$cluster==3)

park_map <- get_googlemap(center = cen1, maptype = "roadmap",
                          zoom = 13, size = c(640,640))
ggmap(park_map) +
  geom_point(data = p1, aes(x=lon, y=lat),
             colour = 'blue', size=2) +
  geom_point(data= p2, aes(x=lon,y=lat),
             colour= "red", size=2) +
  geom_point(data= p3, aes(x=lon,y=lat),
             colour= "darkgreen", size=2)



# 중심좌표와 공원과 거리계산(군집 3개)
m <- data.frame(fit$centers)
mat2 <- distm(m[,c('lon', 'lat')], park[,c('lon', 'lat')])
won2 <- park$공원명[max.col(-mat2)]
table(won2)


# 군집개수 2개
fit1 <- kmeans(x=data1, centers = 2) 
fit1

fit1$cluster
fit1$centers

clusplot(data1, fit1$cluster, color = TRUE, 
         shade = TRUE, labels = 2, lines = 0)

# 지도찍기
point1 <- subset(data1, fit1$cluster==1)
point2 <- subset(data1, fit1$cluster==2)

park_map <- get_googlemap(center = cen, maptype = "roadmap",
                          zoom = 13, size = c(640,640))
ggmap(park_map) +
  geom_point(data = point1, aes(x=lon, y=lat),
             colour = 'blue', size=2) +
  geom_point(data = point2, aes(x=lon,y=lat),
             colour= "red", size=2)


# 중심좌표와 공원과 거리계산(군집 2개)
m2 <- data.frame(fit1$centers)
mat3 <- distm(m2[,c('lon', 'lat')], park[,c('lon', 'lat')])
won3 <- park$공원명[max.col(-mat3)]
table(won3)


install.packages("NbClust")
library(NbClust)

nc <- NbClust(data1, min.nc = 2, max.nc = 15, method = "kmeans")

par(mfrow=c(1,1))
barplot(table(nc$Best.n[1,]))
