FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY tsconfig.json ./
COPY next.config.js ./
COPY postcss.config.js ./
COPY tailwind.config.js ./
COPY src ./src

# NEXT_PUBLIC_ vars are baked in at build time — must be set before npm run build
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ARG NEXT_PUBLIC_API_KEY=flowwatch-dev-key-001
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_API_KEY=$NEXT_PUBLIC_API_KEY

RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
