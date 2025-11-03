# YOLO Mode Usage Examples

## Basic Workflow

1. **Enable YOLO Mode**
   ```
   :yolo
   ```
   Output: "YOLO mode enabled"

2. **Switch to Your Project**
   ```
   :cd C:\Users\randomperson\IdeaProjects\class
   ```
   Output: "✓ Switched to project: C:\Users\randomperson\IdeaProjects\class"

3. **Analyze the Codebase**
   ```
   :analyze
   ```
   This will:
   - Scan all source files
   - Count lines by file type
   - Display project structure
   - **Auto-read 10 key files** (README, package.json, main files, etc.)
   - Inject all code into AI context

4. **Ask Questions!**
   ```
   What does this codebase do?
   Explain the authentication system
   Find all API endpoints
   Review for security issues
   What frameworks are being used?
   ```

## Advanced Commands

### List All Files
```
:files
```
Shows all source files in the project.

### Filter Files
```
:files tsx          # All TypeScript React files
:files api          # All files with "api" in the name
:files auth         # All authentication-related files
```

### Read Specific Files
```
:read src/App.tsx
:read backend/api/login.js
:read package.json
```

### Check Current Project
```
:project
```
Shows current project path and recent projects.

### Switch Projects Quickly
```
:cd ~/myapp
:cd C:\Projects\website
:cd ../other-project
```

## Real-World Example

```
You: :yolo
YOLO mode enabled

You: :cd C:\Users\randomperson\IdeaProjects\class
✓ Switched to project: C:\Users\randomperson\IdeaProjects\class

You: :analyze
📊 Code Analysis
Location: C:\Users\randomperson\IdeaProjects\class
Files: 115
Lines: 29,026

File Types:
  .json: 17,139 lines
  .tsx: 5,245 lines
  .ts: 2,189 lines
  ...

[Project tree displayed]
Reading key files for AI context...
✓ Project context added (10 files analyzed)
You can now ask questions about the codebase!

You: What is this project about?
Assistant: This is a full-stack web application built with:
- Frontend: React + TypeScript + Vite
- Backend: Node.js with Express
- Database: Prisma ORM
- Deployment: GitHub Actions, Vercel/Render

The application appears to be a class/school website with features like:
- Image gallery with lazy loading
- User authentication (login/register)
- Admin panel
- Music player
- Member profiles
- Social media integration

Key architecture:
- `src/` contains React components and pages
- `backend/` contains API endpoints and server logic
- `api/` contains serverless functions for deployment
- Database schema defined in `backend/prisma/schema.prisma`

You: Show me all the API endpoints
Assistant: Based on the code analysis, here are all the API endpoints:

**Authentication:**
- POST /api/login - User login
- POST /api/register - User registration

**Images:**
- GET /api/images - Get all images
- POST /api/upload - Upload new image (requires auth)
- GET /api/images/discord - Get Discord images

**System:**
- GET /api/health - Health check endpoint
- GET /api/cron - Scheduled tasks

[AI continues with detailed explanations from the actual code...]

You: :read backend/api/login.js
[Shows the login.js file content]

You: Review this login code for security issues
Assistant: I've reviewed the login code and found these security concerns:

1. **Password Storage**: ✅ Good - Uses bcrypt for hashing
2. **JWT Security**: ⚠️ Warning - JWT secret should be longer
3. **Rate Limiting**: ❌ Missing - No protection against brute force
4. **Input Validation**: ⚠️ Partial - Could be more comprehensive
...

You: :files auth
Files matching 'auth':
  backend/auth.js
  backend/api/_auth.js
  src/services/api.ts (contains auth functions)
  src/pages/LoginPage.tsx

You: :read backend/auth.js
[Shows auth.js content]

You: How does the JWT token validation work?
Assistant: [Explains with actual code from auth.js...]
```

## Tips

1. **Use :analyze first** - It auto-reads the most important files
2. **Use :files <pattern>** - To discover relevant files before reading them
3. **Use :read** - For files you want to examine in detail
4. **Ask follow-up questions** - The AI retains all file context
5. **Switch projects easily** - Use :cd to analyze multiple codebases

## What Gets Auto-Read by :analyze?

Priority patterns (up to 10 files):
1. README.md
2. package.json, tsconfig.json, vite.config, etc.
3. main.*, index.*, App.*
4. server.*, api.*, config.*
5. schema.*, routes.*, middleware.*
6. Any other important-looking files

This ensures the AI understands:
- Project structure and dependencies
- Entry points and configuration
- Core application logic
- Database schemas
- API routes
