# This file is for preprocessing the raw data of HCP
# load the HCP data
load("HCP_language.Rdata")

# load the module information 
library(readxl)
module=read_excel("Power264_Consensus264.xlsx",sheet=1,na="NA")
module=module[-1,c(1,37)]

# we only consider the nodes inside these four modules
module_num=c()
module_name=c()
for (m in 1:264){
  if ((module[m,2]=="Auditory")|(module[m,2]=="Default mode")|(module[m,2]=="Visual")|(module[m,2]=="Fronto-parietal Task Control")){
    module_num=c(module_num,m)
    module_name=c(module_name,module[m,2])
  }
}

# save the module names
write.csv(module_name, file = "module_name.csv")

# extract sample id that belongs to the sbj_idx object from the whole samples
idx=c()
for (m in 1:928){
  idx=c(idx,(1:1206)[sbj_info[1:1206,1]==sbj_idx[m]])
}

# find the sample id where the performance score is 100, it is man and has age between 22-25
record=c()
for (m in 1:928){
  node=idx[m]
  if((sbj_info[1:1206,21][node]==100)&(sbj_info[1:1206,4][node]=="M")&(sbj_info[1:1206,5][node]=="22-25")){
    record=c(record,m)
  }
}
set.seed(2020)
# randomly select 6 units from the id set
sample_id=sample(1:length(record),6)
record=record[sample_id]
record=sort(record)
length(record)
record_22_25_male=record

# find the sample id where the performance score is 100, it is man and has age between 26-30
record=c()
for (m in 1:928){
  node=idx[m]
  if((sbj_info[1:1206,21][node]==100)&(sbj_info[1:1206,4][node]=="M")&(sbj_info[1:1206,5][node]=="26-30")){
    record=c(record,m)
  }
}
set.seed(2020)
# randomly select 3 units from the id set
sample_id=sample(1:length(record),3)
record=record[sample_id]
record=sort(record)
length(record)
record_26_30_male=record


# find the sample id where the performance score is 100, it is female and has age between 22-25
record=c()
for (m in 1:928){
  node=idx[m]
  if((sbj_info[1:1206,21][node]==100)&(sbj_info[1:1206,4][node]=="F")&(sbj_info[1:1206,5][node]=="22-25")){
    record=c(record,m)
  }
}
set.seed(2020)
# randomly select 3 units from the id set
sample_id=sample(1:length(record),3)
record=record[sample_id]
record=sort(record)
length(record)
record_22_25_female=record

# find the sample id where the performance score is 100, it is female and has age between 26-30
record=c()
for (m in 1:928){
  node=idx[m]
  if((sbj_info[1:1206,21][node]==100)&(sbj_info[1:1206,4][node]=="F")&(sbj_info[1:1206,5][node]=="26-30")){
    record=c(record,m)
  }
}
set.seed(2020)
# randomly select 8 units from the id set
sample_id=sample(1:length(record),8)
record=record[sample_id]
record=sort(record)
length(record)
record_26_30_female=record

# find the sample id where the performance score is 100, it is female and has age between 31-35
record=c()
for (m in 1:928){
  node=idx[m]
  if((sbj_info[1:1206,21][node]==100)&(sbj_info[1:1206,4][node]=="F")&(sbj_info[1:1206,5][node]=="31-35")){
    record=c(record,m)
  }
}
set.seed(2020)
# randomly select 8 units from the id set
sample_id=sample(1:length(record),8)
record=record[sample_id]
record=sort(record)
record_31_35_female=record

# aggregate all the selectd id together as our subjects for the high performance group. In this way, we can make sure that the high performance group has the similar distributions of attributes as the low performance group.
record=c(record_22_25_male,record_26_30_male,record_22_25_female,record_26_30_female,record_31_35_female)
record=sort(record)
length(record)

# extract the time series data for the high performance group
HCP_new=list()
for (m in 1:length(record)){
  content=HCP_data[[record[m]]][module_num,1:316]
  HCP_new[[m]]=content
}
HCP_test_new=matrix(0,length(record)*316,127)
for (m in 1:length(record)){
  HCP_test_new[1:316+(m-1)*316,1:127]=t(HCP_new[[m]][1:127,1:316])
}

# convert the csv to numpy type later by using pandas package for the python codes
write.csv(HCP_test_new, file = "HCP_high.csv")


idx=c()
for (m in 1:928){
  idx=c(idx,(1:1206)[sbj_info[1:1206,1]==sbj_idx[m]])
}

# get the id for the performance score equal or lower than 65 as the low performance group
record=c()
for (m in 1:928){
  node=idx[m]
  if((sbj_info[1:1206,21][node]<=65)){
    record=c(record,m)
  }
}

# extract the time series data for the low performance group
HCP_new=list()
for (m in 1:length(record)){
  content=HCP_data[[record[m]]][module_num,1:316]
  HCP_new[[m]]=content
}
HCP_test_new=matrix(0,length(record)*316,127)
for (m in 1:length(record)){
  HCP_test_new[1:316+(m-1)*316,1:127]=t(HCP_new[[m]][1:127,1:316])
}

# convert the csv to numpy type later by using pandas package for the python codes
write.csv(HCP_test_new, file = "HCP_low.csv")