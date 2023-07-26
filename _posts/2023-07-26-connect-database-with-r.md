---
layout: post
title:  "Connect Database with R"
categories: [Tutorial, R, Database, SQL]
tags: [favicon, r, database, sql]
---



## Set Environment Variables

1.  Before we start running the Rscript, we should set the database
    `username` and `password` as environment variables. You should have
    `username` and `password` from your databse

2.  I create a file `set_env.R` with those lines below, and store it
    somewhere.

3.  <b> REMEMBER: </b>

    SET Environment Variables using <b> `Sys.setenv()`</b> and GET
    Environment Variables <b>`Sys.getenv()`</b>. Don’t use the wrong
    funtion.



```
Sys.setenv("database_username"="your_username",
           "database_password"="your_password")
```


  You can check whether your environment variables already stored by
    `Sys.getenv()`. This is an example.



```
Sys.getenv("database_username")
```


## Connect to Database via R

1.  Load `library(DBI)` and `library(RPostgres)`. If you don’t have
    these two library, please install them.

2.  Connect database by `DBI::dbConnect()` with parameters
    `RPostgres::Postgres()`,`dbname`, `host`, `port`, `user`, and
    `password`



```
library(DBI)
library(RPostgres)
library(dplyr)

# connect to database
con <- DBI::dbConnect(
  RPostgres::Postgres(),
  dbname = 'database_R', 
  host = 'localhost', 
  port = 5432,
  user = Sys.getenv("database_username"), 
  password = Sys.getenv("database_password")
)

con
```

    ## <PqConnection> database_R@localhost:5432



 Using `dbListTables()` to list all of tables in your database.



```
# returns a list of tables in your database
RPostgres::dbListTables(con) 
```

    ## [1] "employee"   "department" "cars"       "iris"



### Create tables via R

1.  Using `CREATE TABLE` like sql query to create a table.

2.  Send the query to database with `RPostgres:: dbSendQuery()`



```
stmt = "CREATE TABLE department (
         department_id int unique not null,
         department varchar (100), 
           department_budget numeric(9,2),
         PRIMARY KEY (department_id)
         );"

RPostgres:: dbSendQuery(con, stmt)
```



  You can see the `department` table existed in the database
    `database_R`



<img src="/assets/img/connect_database/query.png" width="450" />



### Extract, Transform, and Load (ETL) via R to database

#### 1. Extract data



```
# read data
dt <- read.csv("database_R_data.csv")
tibble::glimpse(dt)
```

    ## Rows: 100
    ## Columns: 5
    ## $ employee_id       <int> 2156, 17845, 31201, 39744, 47397, 92164, 133720, 137…
    ## $ first_name        <chr> "Rosmunda", "Carleton", "Jacky", "Fulvia", "Ambrosi"…
    ## $ last_name         <chr> "Grellis", "Buckenham", "Pickin", "Hearnes", "Boule"…
    ## $ department        <chr> "Services", "Engineering", "Services", "Services", "…
    ## $ department_budget <int> 600000, 1250000, 600000, 600000, 750000, 1000000, 45…



#### 2. Transform data



Create department table by select `department, department_budget`, make
sure it’s not duplicated and create `department_id`

```
department <- dt %>% select(department, department_budget) %>% unique() %>% mutate(department_id := 1:n()) %>% select(department_id, everything())

department 
```

    ##    department_id               department department_budget
    ## 1              1                 Services            600000
    ## 2              2              Engineering           1250000
    ## 5              3                Marketing            750000
    ## 6              4 Research and Development           1000000
    ## 7              5          Human Resources            450000
    ## 8              6     Business Development            200000
    ## 9              7                    Legal            350000
    ## 10             8                  Support            500000
    ## 13             9               Accounting            500000
    ## 14            10                 Training            400000
    ## 18            11       Product Management           2000000
    ## 21            12                    Sales           1100000



#### 3. Load data to database



You can use `dbWriteTable()` or `RPostgres::dbSendQuery()`

```
# you can use dbWriteTable
RPostgres::dbWriteTable(con,'department',department , row.names=FALSE,overwrite = TRUE)


# or use these codes
RPostgres::dbSendQuery(
  con, 
  "INSERT INTO department (department_id ,department,department_budget) VALUES ($1,$2,$3);",
  list(
    department$department_id,
    department$department,
    department$department_budget
  )
)
```



#### 4. Checking the table existed in database



```
checking_department <- RPostgres::dbGetQuery(con, 'SELECT * FROM department')
checking_department 
```

    ##    department_id               department department_budget
    ## 1              1               Accounting            500000
    ## 2              2                    Sales           1100000
    ## 3              3                Marketing            750000
    ## 4              4          Human Resources            450000
    ## 5              5              Engineering           1250000
    ## 6              6 Research and Development           1000000
    ## 7              7       Product Management           2000000
    ## 8              8                 Services            600000
    ## 9              9                    Legal            350000
    ## 10            10                 Training            400000
    ## 11            11                  Support            500000
    ## 12            12     Business Development            200000



<img src="/assets/img/connect_database/checking.png" width="450" />



#### 5. Uploading any dataset to your database



```
# upload iris dataset into the database
RPostgres::dbWriteTable(con,'iris',iris , row.names=FALSE,overwrite = TRUE)

# checking iris dataset exist in your database
iris <- RPostgres::dbGetQuery(con, 'SELECT * FROM iris')
tibble::glimpse(iris)
```

    ## Rows: 150
    ## Columns: 5
    ## $ Sepal.Length <dbl> 5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9, 5.4, 4.…
    ## $ Sepal.Width  <dbl> 3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1, 3.7, 3.…
    ## $ Petal.Length <dbl> 1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5, 1.5, 1.…
    ## $ Petal.Width  <dbl> 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.2, 0.…
    ## $ Species      <chr> "setosa", "setosa", "setosa", "setosa", "setosa", "setosa…



<img src="/assets/img/connect_database/iris.png" width="450" />



<center>
<h3>
Thank you for reading
</h3>
</center>
