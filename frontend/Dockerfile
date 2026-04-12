FROM node:24.13-alpine3.22 AS base
WORKDIR /usr/src/app
RUN chown node:node .
COPY --chown=node:node package.json package-lock.json ./
USER node

FROM base AS development
ENV HOST=0.0.0.0
RUN npm ci && npm cache clean --force
CMD ["npm", "run", "dev", "--", "--host"]

FROM base AS builder
COPY --chown=node:node . .
RUN npm ci && npm cache clean --force
RUN npm run build

# a test
FROM nginx:1.29-alpine AS production 
RUN rm -rf /usr/share/nginx/html/*
COPY --from=builder /usr/src/app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
CMD ["nginx", "-g", "daemon off;"]