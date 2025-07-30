# Detector de Cor de Objetos - Deploy Gratuito

## 📱 Como rodar ao vivo no celular (com frontend)

### 1. Backend (API Flask)

Você precisa implantar seu backend Flask em um serviço gratuito que aceite Python. As melhores opções são:

- **[Render.com](https://render.com/):**  
  Permite rodar Flask de graça (Web Service).  
  - Suba seu projeto para o GitHub.
  - No Render, clique em "New Web Service", conecte seu repositório e use:
    - Build Command: `pip install -r requirements.txt`
    - Start Command: `python src/app.py`
  - Após o deploy, você terá uma URL pública para sua API.

- **[Railway.app](https://railway.app/):**  
  Também aceita Flask, fácil de usar, plano gratuito.

- **[Fly.io](https://fly.io/):**  
  Aceita Docker, plano gratuito, pode rodar Flask.

> **Importante:** O backend precisa estar acessível publicamente para o frontend funcionar no celular.

---

### 2. Frontend (HTML/JS)

- Hospede o arquivo `index.html` em um serviço gratuito de sites estáticos:
  - [Netlify](https://netlify.com/)
  - [Vercel](https://vercel.com/)
  - [GitHub Pages](https://pages.github.com/)

No arquivo `index.html`, troque a variável `API_URL` pela URL do seu backend Flask.

---

### 3. Como usar

1. Abra o site frontend no navegador do celular.
2. Permita o acesso à câmera.
3. O site vai capturar frames e enviar para o backend, mostrando o resultado ao vivo.

---

### 4. Dicas

- O backend Flask não roda direto no Netlify ou Vercel, use Render, Railway ou Fly.io.
- O frontend pode ser hospedado em qualquer serviço de site estático.
- Para melhor desempenho, envie imagens menores (redimensione no frontend antes de enviar).

---

**Dúvidas? Só perguntar!**