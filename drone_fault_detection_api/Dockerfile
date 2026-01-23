# Node.js API Dockerfile
FROM node:18-alpine

WORKDIR /app

# Package dosyalarını kopyala
COPY package*.json ./

# Bağımlılıkları yükle
RUN npm install --production

# Kaynak kodları kopyala
COPY . .

# Uploads klasörünü oluştur
RUN mkdir -p uploads

# Port
EXPOSE 3001

# Başlat
CMD ["npm", "start"]
