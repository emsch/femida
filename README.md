# femida
Фемида | Система распознавания тестов ЭМШ


# Getting Started

* Pull the repository
* run the following (in the project root)
```
cat Makefile.template > Makefile
cat envfile.template > envfile
cat secret.template > secret
```
* make changes in env files if required
* run `docker-compose build`
* find or train a ocr model and place it to `./model/model.t7`
* run `docker-compose up`
