# backend/Dockerfile
FROM node:20-alpine

WORKDIR /app

# Install sharp dependencies
RUN apk add --no-cache vips-dev fftw-dev build-base python3

COPY package*.json ./
RUN npm ci --only=production

COPY . .

# Create uploads directory
RUN mkdir -p uploads && chmod 755 uploads

EXPOSE 3001

CMD ["node", "server.js"]
