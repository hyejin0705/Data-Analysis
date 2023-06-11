## 팝콘 - 청소년 팝업쉼터
# install.packages("readr")
# install.packages("readxl")
# install.packages("dplyr")
# install.packages("ggplot2")
library(readr)
library(readxl)
library(dplyr)
library(ggplot2)

## 청소년들의 위험신호

## 0~14 VS 19세 이하 자살 수 비교
# 그냥 lineplot
teen <- read_excel("자살수_0~14세.xlsx")
year <- c(teen$기간)
per <- c(teen$계)
teen2 <- read_excel("자살수_15~19세.xlsx")
year2 <- c(teen2$기간)
per2 <- c(teen2$계)

plot(year,per, main='0~19세 사망자수(명)',type='o',lty=1,lwd=5,
     xlab='year',ylab='사망자수', col="orange",ylim=c(0,60))
lines(year2,per2, type='o',lty=1,lwd=5,
      xlab='year',ylab='사망자수', col="red")

# ggplot 사용
teen <- read_excel("19세_자살_비교.xlsx")

# teen$기간이 문자형이기 때문에 숫자형으로 바꾸고 x축으로 넣어줘야 한다.
ggplot(data = teen, 
       aes(x= as.numeric(기간), y=계, color=연령)) +
  geom_line() +
  geom_point() +
  ggtitle("0~19세 자살 수(명) 비교") +
  xlab("자살사망수") + ylab("연도")


## 청소년 연도별 사망원인

# 데이터 행렬 바꾸는 패키지(reshape2)
install.packages("reshape2")
library(reshape2)
reason <- read_excel("청소년 사망원인.xlsx")
head(reason)

# melt(데이터, id.var=) - 열을 행으로 바꾸는 명령어 
# id.vars = c(기준 열)
data_melt <- melt(reason, id.vars = c('연도'),
                  variable.name="사망원인", 
                  value.name='value')  # 열 이름 변경
head(data_melt)

ggplot(data = data_melt, 
       aes(x= 연도, y=value, color=사망원인)) +
  geom_line() +
  geom_point() +
  scale_x_continuous(breaks=seq(2010, 2019)) +
  theme_bw() +
  ggtitle("청소년 연도별 사망원인") +
  ylab("")

# scale_x_continuous(breaks=seq(2010, 2019)) - x축 눈금 설정
# theme_bw() - 흰 배경


## 코로나19 이후 삶의 변화
co <- read_excel("코로나19 이후 삶의 변화.xlsx")

data_co <- melt(co, id.vars=c('구분'),
                variable.name="변화", 
                value.name='value')

# 막대그래프
ggplot(data = data_co, 
       aes(x=factor(구분), y=value, fill=변화)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_bw() +
  ggtitle("청소년들이 느끼는 코로나19 이후 삶의 변화") +
  xlab("영역") + ylab("프로(%)")

# factor() 범주형(카테고리 형식)으로 만들어줘
# position = "dodge" 그래프를 옆으로 병렬해줘!


## 2017~2019 성인 VS 청소년 스트레스인지율 비교
ad <- read.csv("스트레스.csv")
ad

# 상자그림
# reshape2 - dcast(데이터, 기준, 변환) 행을 열로 변경!
ad_col <- dcast(ad, 연도 ~ 구별)
ad_col
# Using 스트레스 as value column: use value.var to override.
# - 에러메시지 아니고 정상적으로 변환되었음을 의미.

describe(ad_col$성인)
describe(ad_col$청소년)
boxplot(ad_col$성인, ad_col$청소년, 
        main = "성인, 청소년 스트레스 비교",
        names = c("성인", "청소년"),
        ylab = "스트레스", col = c("blue", "orange"))


# T검정 패키지 psych
# install.packages("psych")
library(psych)
t.test(data=ad, 스트레스~구별, var.equal=T)


# 위와 동일
## 2017~2019 청소년 남학생 VS 여학생 스트레스인지율 비교
young <- read.csv("청소년 남녀.csv")

# 상자그림
young_col <- dcast(young, 연도 ~ 성별)
describe(young_col$남학생)
describe(young_col$여학생)
boxplot(young_col$남학생, young_col$여학생, 
        main = "남학생, 여학생 스트레스 비교",
        names = c("남학생", "여학생"),
        ylab = "스트레스", col = c("red", "orange"))

# T검정
t.test(data=young, 스트레스~성별, var.equal=T)


## 청소년 정신건강 진료
library(ggplot2)
mind <- read_excel("아동_청소년의 정신진료 현황.xlsx")

# x축이 문자형인데 선 하나를 그릴 경우, group = 1 사용
ggplot(mind, aes(x = 구분, y = `10~19세`, group = 1)) +
  geom_line(color="#5882FA",size=2)+
  ggtitle("정신건강진료") +
  theme(plot.title = element_text(size = 25, face = "bold", colour="#424242"))+
  labs(x="연령", y="정신건강 진료")


## 상담대기일수
wait <- read_excel("청소년상담복지센터 인력과 대기일수.xlsx")

# reorder 를 쓰면 오름차순으로 정렬, '-' 를 붙이면 내림차순
wait %>% ggplot(aes(x=reorder(구분, -`상담대기(일)`),
                    y=`상담대기(일)`, fill = `상담대기(일)`)) +
  ggtitle("상담대기일수") +
  xlab('지역') + ylab('기간') +
  geom_bar(stat="identity") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
# 지역이름이 많아서 겹침. theme을 추가하여 x_label을 45도 회전

