<<<<<<< HEAD

## Overview

This project builds an integrated AI Agent ecosystem using the Model Context Protocol (MCP). The agent is deployed locally using Docker and n8n, exposed to the internet via ngrok, and connected to Telegram and Google Calendar for real-world task automation.

---

## System Architecture

```
Telegram User
     |
     v
  ngrok (public URL)
     |
     v
  n8n (localhost:5678)
     |
     |-- AI Agent Workflow
     |       |-- Groq LLM (llama-3.3-70b-versatile)
     |       |-- Postgres Chat Memory
     |       |-- MCP Client --> MCP Server Workflow
     |       |                      |-- Calculator Tool
     |       |                      |-- Date & Time Tool
     |       |                      |-- Code Tool
     |       |-- Google Calendar Tool
     |       |-- Telegram Reply
     |
     v
  PostgreSQL Database (Docker)
```

---

## Requirements

- Docker Desktop
- ngrok (free account)
- Groq API key (free at https://console.groq.com)
- Telegram account and bot token (via BotFather)
- Google account with Calendar API enabled

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd A7
```

### 2. Create the .env File

Copy the example file and fill in your own values:

```bash
cp .env.example .env
```

Open `.env` and fill in:

```
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_NAME=your_db_name
NGROK_URL=your_ngrok_url
```

### 3. Start ngrok

Open a terminal and run:

```bash
ngrok http 5678
```

Copy the forwarding URL (e.g. `https://xxxx.ngrok-free.dev`) and paste it as the `NGROK_URL` value in your `.env` file.

### 4. Start Docker Containers

```bash
docker compose up -d
```

This starts two containers:
- `a7-n8n-1` — the n8n automation platform (accessible at http://localhost:5678)
- `a7-db-1` — PostgreSQL database for n8n storage

### 5. Open n8n

Go to http://localhost:5678 in your browser and log in.

---

## n8n Workflow Setup

### MCP Server Workflow

Create a workflow named **MCP Server** with the following nodes:
- MCP Server Trigger
- Calculator (Tool)
- Date and Time (Tool)
- Code Tool (Tool)

Publish the workflow and copy the Production URL.

### AI Agent Workflow

Create a workflow named **AI Agent** with the following nodes:
- When chat message received (Trigger)
- Telegram Trigger (On message)
- AI Agent
  - Groq Chat Model (llama-3.3-70b-versatile)
  - Postgres Chat Memory
  - MCP Client (connected to MCP Server Production URL)
  - Google Calendar Tool
- Telegram Send (Send a text message)

Publish the workflow.

---

## Environment Variables

| Variable | Description |
|---|---|
| `DB_USER` | PostgreSQL username |
| `DB_PASSWORD` | PostgreSQL password |
| `DB_NAME` | PostgreSQL database name |
| `NGROK_URL` | Your ngrok public forwarding URL |

---

## Tools and Technologies

| Tool | Purpose |
|---|---|
| n8n | Workflow automation platform |
| Docker | Container deployment |
| ngrok | Expose localhost to the internet |
| Groq API | Free LLM provider (llama-3.3-70b-versatile) |
| Telegram Bot API | Messaging interface |
| Google Calendar API | Calendar management |
| PostgreSQL | Database for n8n and chat memory |
| MCP (Model Context Protocol) | Tool integration protocol |

---

## Notes

- The ngrok URL changes every time you restart ngrok on the free plan. Update your `.env` file and restart n8n whenever this happens using `docker compose restart n8n`.
- The MCP Server workflow must be running (Execute workflow) for the AI Agent to use MCP tools.
- Make sure the Telegram webhook is registered with the correct ngrok URL.
=======
This repository contains all of my Natural Language Processing (NLP) assignments for my university course.

I will be uploading all my NLP assignments here throughout the semester. Each assignment will include the required code, explanations, and any necessary files.

This repository is created for academic purposes only.
>>>>>>> 293e6f27d2752572f0a4d5cd7dc24065acd16cee
