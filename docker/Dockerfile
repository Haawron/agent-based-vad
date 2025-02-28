# DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile -t torch --build-arg UID=$(id -u) --build-arg GID=$(id -g) .

FROM ubuntu:22.04 AS base

ARG UID
ARG GID
ARG USERNAME=hglee

RUN groupadd -g ${GID} ${USERNAME} && \
    useradd -m -u ${UID} -g ${USERNAME} ${USERNAME}

# ----------------------------------------------------------------------

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ARG USERNAME=hglee

# Copy only the minimal needed configuration or files.
COPY --from=base /etc/passwd /etc/passwd
COPY --from=base /etc/group /etc/group
RUN mkdir -p /home/${USERNAME} && chown ${USERNAME}:${USERNAME} /home/${USERNAME}
# Add user to sudo group
RUN usermod -aG sudo ${USERNAME}

RUN sed -i 's@archive.ubuntu.com@mirror.kakao.com@g' /etc/apt/sources.list

ENV TZ=Asia/Seoul
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y \
        vim git zsh htop iftop tmux unzip zip curl wget tree ncdu \
        tzdata bzip2 build-essential net-tools iputils-ping locales && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean
RUN curl -fsSL https://starship.rs/install.sh | sh -s -- -y

RUN locale-gen en_US.UTF-8 && \
    locale-gen ko_KR.UTF-8 && \
    update-locale LANG=en_US.UTF-8


# Install python packages
RUN pip install transformers==4.48.3 sgl-kernel --force-reinstall --no-deps && \
    pip install "sglang[all]>=0.4.3.post2" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

# ----------------------------------------------------------------------
RUN chsh -s /usr/bin/zsh ${USERNAME}
USER ${USERNAME}

# Install and configure Zsh and plugins
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.2.1/zsh-in-docker.sh)" -- \
    -t '' \
    -p git \
    -p dotenv \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions \
    -p https://github.com/zsh-users/zsh-syntax-highlighting
# Put export ZSH_DOTENV_PROMPT=false on top of ~/.zshrc
RUN sed -i '1s/^/export ZSH_DOTENV_PROMPT=false\n/' ~/.zshrc

RUN echo 'eval "$(starship init zsh)"' >> ~/.zshrc && \
    mkdir ~/.config && \
    starship preset pastel-powerline -o ~/.config/starship.toml

WORKDIR /code

# login to huggingface
COPY --chmod=0644 .env .

# RUN python -c "import os; from dotenv import load_dotenv; load_dotenv(); from huggingface_hub._login import _login; _login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)"

CMD ["zsh"]
