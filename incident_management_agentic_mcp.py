"""
LLM Orchestrator System with Agentic MCP Servers

Architecture:
1. Orchestrator LLM - Reads from DB, intelligently routes to MCP servers
2. Jira Agentic MCP Server - LLM-powered Jira operations (port 8001)
3. Slack Agentic MCP Server - LLM-powered Slack operations (port 8002)

Each MCP server has its own LLM that makes decisions before calling mock MCP tools.
"""

import asyncio
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
import httpx
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager


# ============================================================================
# SHARED DATA MODELS
# ============================================================================

class IncidentSeverity(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class IncidentRecord(BaseModel):
    """Incident from database"""
    id: str
    title: str
    description: str
    severity: IncidentSeverity
    service: str
    error_rate: float
    affected_users: int
    customer_facing: bool
    revenue_impacting: bool
    detected_at: str
    region: str


# ============================================================================
# MOCK DATABASE
# ============================================================================

class MockDatabase:
    """Simulated incident database"""
    
    def __init__(self):
        self.incidents = [
            {
                "id": "INC-001",
                "title": "Payment Gateway Complete Outage",
                "description": "All payment processing down. 503 errors. Database connection timeouts.",
                "severity": "Critical",
                "service": "payment-gateway",
                "error_rate": 1.0,
                "affected_users": 5000,
                "customer_facing": True,
                "revenue_impacting": True,
                "detected_at": datetime.now().isoformat(),
                "region": "us-east-1"
            },
            {
                "id": "INC-002",
                "title": "Internal Dashboard Slow",
                "description": "Analytics dashboard taking 10+ seconds to load. Internal tool only.",
                "severity": "Medium",
                "service": "analytics-dashboard",
                "error_rate": 0.0,
                "affected_users": 20,
                "customer_facing": False,
                "revenue_impacting": False,
                "detected_at": datetime.now().isoformat(),
                "region": "us-west-2"
            },
            {
                "id": "INC-003",
                "title": "Authentication Service Degraded",
                "description": "Login attempts taking 5-8 seconds. 30% timeout rate during peak hours.",
                "severity": "High",
                "service": "auth-service",
                "error_rate": 0.3,
                "affected_users": 2500,
                "customer_facing": True,
                "revenue_impacting": True,
                "detected_at": datetime.now().isoformat(),
                "region": "eu-west-1"
            }
        ]
    
    async def get_pending_incidents(self) -> List[IncidentRecord]:
        """Fetch pending incidents from DB"""
        await asyncio.sleep(0.1)  # Simulate DB query
        return [IncidentRecord(**inc) for inc in self.incidents]


# ============================================================================
# LLM CLIENT (Shared)
# ============================================================================

class LLMClient:
    """LLM client for Ollama"""
    
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    async def query(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Query LLM"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result["message"]["content"]
            except Exception as e:
                print(f"⚠️  LLM unavailable: {e}")
                return self._mock_response(prompt)
    
    def _mock_response(self, prompt: str) -> str:
        """Mock LLM response when Ollama unavailable"""
        if "orchestrator" in prompt.lower() or "route" in prompt.lower():
            return """```json
{
  "routing_decision": {
    "use_jira": true,
    "use_slack": true,
    "use_pagerduty": false
  },
  "jira_priority": "Critical",
  "slack_channel": "#incidents-critical",
  "create_war_room": true,
  "reasoning": "Critical customer-facing incident with revenue impact requires immediate ticket and team notification"
}
```"""
        elif "jira" in prompt.lower():
            return """```json
{
  "priority": "Critical",
  "assignee": "alice",
  "labels": ["incident", "payment", "critical"],
  "components": ["payment-gateway", "database"],
  "escalate": true,
  "reasoning": "Payment outage requires immediate attention from backend specialist Alice"
}
```"""
        else:  # Slack
            return """```json
{
  "channel_name": "incident-inc-001",
  "notify_users": ["@alice", "@on-call"],
  "message_urgency": "critical",
  "pin_message": true,
  "reasoning": "Critical incident requires dedicated channel with immediate notifications"
}
```"""


# ============================================================================
# JIRA AGENTIC MCP SERVER (Port 8001)
# ============================================================================

jira_app = FastAPI(title="Jira Agentic MCP Server")
jira_llm = LLMClient()


class JiraTicketRequest(BaseModel):
    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    context: Dict[str, Any]


class JiraTicketResponse(BaseModel):
    status: str
    ticket_key: str
    url: str
    priority: str
    assignee: str
    labels: List[str]
    llm_decision: Dict[str, Any]
    llm_reasoning: str


class MockJiraMCP:
    """Mock Jira MCP server calls"""
    
    @staticmethod
    async def create_issue(project: str, summary: str, description: str, 
                          priority: str, assignee: str) -> Dict[str, Any]:
        """Simulate Jira MCP create_issue call"""
        await asyncio.sleep(0.2)  # Simulate API call
        ticket_id = f"JIRA-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return {
            "id": ticket_id,
            "key": ticket_id,
            "self": f"https://jira.company.com/rest/api/2/issue/{ticket_id}",
            "fields": {
                "summary": summary,
                "priority": {"name": priority},
                "assignee": {"name": assignee}
            }
        }


@jira_app.post("/create_ticket", response_model=JiraTicketResponse)
async def create_ticket(request: JiraTicketRequest):
    """
    Jira Agentic MCP Endpoint
    LLM analyzes incident and decides ticket properties before calling mock MCP
    """
    
    print(f"\n{'='*80}")
    print(f"🎫 JIRA AGENTIC MCP SERVER - Processing {request.incident_id}")
    print(f"{'='*80}")
    
    # Step 1: LLM analyzes and makes decisions
    system_prompt = """You are an expert Jira ticket manager with deep SRE experience.
You analyze incidents and make intelligent decisions about ticket properties.
You understand priority levels, team skills, and escalation needs."""
    
    prompt = f"""Analyze this incident and decide optimal Jira ticket properties:

INCIDENT:
- ID: {request.incident_id}
- Title: {request.title}
- Severity: {request.severity}
- Description: {request.description}

CONTEXT:
{json.dumps(request.context, indent=2)}

TEAM:
- alice: backend, database, payment systems (workload: 3)
- bob: frontend, UI (workload: 2)  
- charlie: devops, infrastructure (workload: 4)
- diana: security, backend (workload: 1)

DECIDE:
1. Priority (Critical/High/Medium/Low)
2. Best assignee based on skills and workload
3. Relevant labels
4. Affected components
5. Should this be escalated?

OUTPUT EXACTLY THIS JSON:
```json
{{
  "priority": "Critical|High|Medium|Low",
  "assignee": "alice|bob|charlie|diana",
  "labels": ["label1", "label2"],
  "components": ["component1", "component2"],
  "escalate": true|false,
  "reasoning": "explain your decisions"
}}
```"""

    print(f"🧠 Querying Jira LLM for analysis...")
    llm_response = await jira_llm.query(prompt, system_prompt)
    
    # Parse LLM decision
    try:
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', llm_response, re.DOTALL)
        if json_match:
            llm_decision = json.loads(json_match.group(1))
        else:
            llm_decision = {
                "priority": "High",
                "assignee": "alice",
                "labels": ["incident"],
                "components": ["general"],
                "escalate": False,
                "reasoning": "Using defaults"
            }
    except Exception as e:
        print(f"⚠️  Failed to parse LLM: {e}")
        llm_decision = {
            "priority": "High",
            "assignee": "alice", 
            "labels": ["incident"],
            "components": ["general"],
            "escalate": False,
            "reasoning": f"Parse error: {e}"
        }
    
    print(f"✅ LLM Decision:")
    print(f"   Priority: {llm_decision['priority']}")
    print(f"   Assignee: {llm_decision['assignee']}")
    print(f"   Escalate: {llm_decision['escalate']}")
    print(f"   Reasoning: {llm_decision['reasoning'][:100]}...")
    
    # Step 2: Call Mock Jira MCP with LLM's decisions
    print(f"\n📞 Calling Mock Jira MCP server...")
    mcp_response = await MockJiraMCP.create_issue(
        project="INCIDENT",
        summary=request.title,
        description=request.description,
        priority=llm_decision["priority"],
        assignee=llm_decision["assignee"]
    )
    
    print(f"✅ Mock Jira MCP Response: {mcp_response['key']}")
    
    return JiraTicketResponse(
        status="success",
        ticket_key=mcp_response["key"],
        url=f"https://jira.company.com/browse/{mcp_response['key']}",
        priority=llm_decision["priority"],
        assignee=llm_decision["assignee"],
        labels=llm_decision["labels"],
        llm_decision=llm_decision,
        llm_reasoning=llm_decision["reasoning"]
    )


@jira_app.get("/")
async def jira_root():
    return {
        "service": "Jira Agentic MCP Server",
        "port": 8001,
        "capabilities": ["create_ticket with LLM analysis"],
        "llm_powered": True
    }


# ============================================================================
# SLACK AGENTIC MCP SERVER (Port 8002)
# ============================================================================

slack_app = FastAPI(title="Slack Agentic MCP Server")
slack_llm = LLMClient()


class SlackChannelRequest(BaseModel):
    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    context: Dict[str, Any]


class SlackChannelResponse(BaseModel):
    status: str
    channel_name: str
    channel_id: str
    notify_users: List[str]
    message_sent: bool
    llm_decision: Dict[str, Any]
    llm_reasoning: str


class MockSlackMCP:
    """Mock Slack MCP server calls"""
    
    @staticmethod
    async def create_channel(name: str, topic: str) -> Dict[str, Any]:
        """Simulate Slack MCP conversations.create call"""
        await asyncio.sleep(0.2)
        channel_id = f"C{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return {
            "ok": True,
            "channel": {
                "id": channel_id,
                "name": name,
                "topic": {"value": topic}
            }
        }
    
    @staticmethod
    async def post_message(channel: str, text: str, users: List[str]) -> Dict[str, Any]:
        """Simulate Slack MCP chat.postMessage call"""
        await asyncio.sleep(0.1)
        
        return {
            "ok": True,
            "ts": datetime.now().isoformat(),
            "channel": channel,
            "message": {"text": text}
        }


@slack_app.post("/create_channel", response_model=SlackChannelResponse)
async def create_channel(request: SlackChannelRequest):
    """
    Slack Agentic MCP Endpoint
    LLM analyzes incident and decides channel properties before calling mock MCP
    """
    
    print(f"\n{'='*80}")
    print(f"💬 SLACK AGENTIC MCP SERVER - Processing {request.incident_id}")
    print(f"{'='*80}")
    
    # Step 1: LLM analyzes and makes decisions
    system_prompt = """You are an expert incident communication coordinator.
You decide how to create and configure Slack channels for incidents.
You know when to notify specific people and how urgent the messaging should be."""
    
    prompt = f"""Analyze this incident and decide Slack channel configuration:

INCIDENT:
- ID: {request.incident_id}
- Title: {request.title}
- Severity: {request.severity}
- Description: {request.description}

CONTEXT:
{json.dumps(request.context, indent=2)}

TEAM:
- @alice: Backend lead
- @bob: Frontend lead (currently on-call)
- @charlie: DevOps lead
- @diana: Security lead

DECIDE:
1. Channel name (use incident ID)
2. Who should be notified? (@mentions)
3. Message urgency level
4. Should message be pinned?
5. Initial message content

OUTPUT EXACTLY THIS JSON:
```json
{{
  "channel_name": "incident-xxx",
  "notify_users": ["@user1", "@user2"],
  "message_urgency": "critical|high|medium|low",
  "pin_message": true|false,
  "reasoning": "explain your decisions"
}}
```"""

    print(f"🧠 Querying Slack LLM for analysis...")
    llm_response = await slack_llm.query(prompt, system_prompt)
    
    # Parse LLM decision
    try:
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', llm_response, re.DOTALL)
        if json_match:
            llm_decision = json.loads(json_match.group(1))
        else:
            llm_decision = {
                "channel_name": f"incident-{request.incident_id.lower()}",
                "notify_users": ["@on-call"],
                "message_urgency": "high",
                "pin_message": True,
                "reasoning": "Using defaults"
            }
    except Exception as e:
        print(f"⚠️  Failed to parse LLM: {e}")
        llm_decision = {
            "channel_name": f"incident-{request.incident_id.lower()}",
            "notify_users": ["@on-call"],
            "message_urgency": "high",
            "pin_message": True,
            "reasoning": f"Parse error: {e}"
        }
    
    print(f"✅ LLM Decision:")
    print(f"   Channel: #{llm_decision['channel_name']}")
    print(f"   Notify: {', '.join(llm_decision['notify_users'])}")
    print(f"   Urgency: {llm_decision['message_urgency']}")
    print(f"   Reasoning: {llm_decision['reasoning'][:100]}...")
    
    # Step 2: Call Mock Slack MCP with LLM's decisions
    print(f"\n📞 Calling Mock Slack MCP server...")
    
    # Create channel
    channel_response = await MockSlackMCP.create_channel(
        name=llm_decision["channel_name"],
        topic=f"{request.title} - {request.severity}"
    )
    
    # Post initial message
    urgency_emoji = {
        "critical": "🚨",
        "high": "⚠️",
        "medium": "📢",
        "low": "ℹ️"
    }
    
    message = f"{urgency_emoji.get(llm_decision['message_urgency'], '📢')} **INCIDENT: {request.title}**\n\n"
    message += f"Severity: {request.severity}\n"
    message += f"Service: {request.context.get('service', 'unknown')}\n\n"
    message += f"{request.description}\n\n"
    message += f"Notify: {' '.join(llm_decision['notify_users'])}"
    
    message_response = await MockSlackMCP.post_message(
        channel=channel_response["channel"]["id"],
        text=message,
        users=llm_decision["notify_users"]
    )
    
    print(f"✅ Mock Slack MCP Response: #{llm_decision['channel_name']}")
    
    return SlackChannelResponse(
        status="success",
        channel_name=llm_decision["channel_name"],
        channel_id=channel_response["channel"]["id"],
        notify_users=llm_decision["notify_users"],
        message_sent=message_response["ok"],
        llm_decision=llm_decision,
        llm_reasoning=llm_decision["reasoning"]
    )


@slack_app.get("/")
async def slack_root():
    return {
        "service": "Slack Agentic MCP Server",
        "port": 8002,
        "capabilities": ["create_channel with LLM analysis"],
        "llm_powered": True
    }


# ============================================================================
# ORCHESTRATOR LLM (Main Server - Port 8000)
# ============================================================================

orchestrator_app = FastAPI(title="LLM Orchestrator")
orchestrator_llm = LLMClient()
mock_db = MockDatabase()


class OrchestrationResult(BaseModel):
    incident_id: str
    routing_decision: Dict[str, Any]
    jira_result: Optional[Dict[str, Any]]
    slack_result: Optional[Dict[str, Any]]
    orchestrator_reasoning: str
    total_time_ms: float


@orchestrator_app.get("/")
async def orchestrator_root():
    return {
        "service": "LLM Orchestrator",
        "port": 8000,
        "description": "Intelligently routes incidents to Jira/Slack MCP servers",
        "architecture": {
            "orchestrator": "http://localhost:8000 (this server)",
            "jira_mcp": "http://localhost:8001",
            "slack_mcp": "http://localhost:8002"
        },
        "endpoints": {
            "POST /process": "Process all pending incidents",
            "GET /health": "Health check all services"
        }
    }


@orchestrator_app.post("/process", response_model=List[OrchestrationResult])
async def process_incidents():
    """
    Main orchestration endpoint:
    1. Reads incidents from DB
    2. LLM decides routing strategy
    3. Calls appropriate MCP servers via HTTP
    """
    
    print(f"\n{'='*80}")
    print(f"🎭 ORCHESTRATOR LLM - Starting Incident Processing")
    print(f"{'='*80}\n")
    
    # Step 1: Read from database
    print(f"📊 Reading incidents from database...")
    incidents = await mock_db.get_pending_incidents()
    print(f"✅ Found {len(incidents)} pending incidents\n")
    
    results = []
    
    for incident in incidents:
        start_time = datetime.now()
        
        print(f"\n{'='*80}")
        print(f"Processing: {incident.id} - {incident.title}")
        print(f"{'='*80}")
        
        # Step 2: LLM decides routing
        system_prompt = """You are an expert incident orchestrator.
You analyze incidents and intelligently decide which systems to notify.
Consider severity, business impact, and appropriate response channels."""
        
        prompt = f"""Analyze this incident and decide routing strategy:

INCIDENT:
{incident.model_dump_json(indent=2)}

AVAILABLE MCP SERVERS:
1. Jira MCP (port 8001) - Create tickets, assign engineers
2. Slack MCP (port 8002) - Create channels, notify teams

DECISION FACTORS:
- Severity level
- Customer/revenue impact  
- Error rate and affected users
- Appropriate response time
- Communication needs

DECIDE:
1. Should we create Jira ticket? (always yes for trackability)
2. Should we create Slack channel? (based on severity)
3. What priority for Jira?
4. Which Slack channel? (#incidents vs #incidents-critical)
5. Should we create dedicated war room?

OUTPUT EXACTLY THIS JSON:
```json
{{
  "routing_decision": {{
    "use_jira": true|false,
    "use_slack": true|false,
    "use_pagerduty": true|false
  }},
  "jira_priority": "Critical|High|Medium|Low",
  "slack_channel": "#incidents-critical|#incidents|#incidents-low",
  "create_war_room": true|false,
  "reasoning": "explain routing decisions"
}}
```"""

        print(f"\n🧠 Orchestrator LLM deciding routing...")
        llm_response = await orchestrator_llm.query(prompt, system_prompt)
        
        # Parse routing decision
        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', llm_response, re.DOTALL)
            if json_match:
                routing = json.loads(json_match.group(1))
            else:
                routing = {
                    "routing_decision": {"use_jira": True, "use_slack": True, "use_pagerduty": False},
                    "jira_priority": "High",
                    "slack_channel": "#incidents",
                    "create_war_room": False,
                    "reasoning": "Using defaults"
                }
        except Exception as e:
            print(f"⚠️  Failed to parse LLM: {e}")
            routing = {
                "routing_decision": {"use_jira": True, "use_slack": True, "use_pagerduty": False},
                "jira_priority": "High",
                "slack_channel": "#incidents",
                "create_war_room": False,
                "reasoning": f"Parse error: {e}"
            }
        
        print(f"✅ Routing Decision:")
        print(f"   Jira: {routing['routing_decision']['use_jira']}")
        print(f"   Slack: {routing['routing_decision']['use_slack']}")
        print(f"   War Room: {routing.get('create_war_room', False)}")
        print(f"   Reasoning: {routing['reasoning'][:100]}...")
        
        jira_result = None
        slack_result = None
        
        # Step 3: Route to Jira MCP (HTTP call)
        if routing['routing_decision']['use_jira']:
            print(f"\n🔀 Routing to Jira Agentic MCP Server (HTTP)...")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    response = await client.post(
                        "http://localhost:8001/create_ticket",
                        json={
                            "incident_id": incident.id,
                            "title": incident.title,
                            "description": incident.description,
                            "severity": incident.severity,
                            "context": {
                                "service": incident.service,
                                "error_rate": incident.error_rate,
                                "affected_users": incident.affected_users,
                                "customer_facing": incident.customer_facing,
                                "revenue_impacting": incident.revenue_impacting,
                                "region": incident.region
                            }
                        }
                    )
                    jira_result = response.json()
                    print(f"✅ Jira ticket created: {jira_result['ticket_key']}")
                except Exception as e:
                    print(f"❌ Jira MCP error: {e}")
                    jira_result = {"error": str(e)}
        
        # Step 4: Route to Slack MCP (HTTP call)
        if routing['routing_decision']['use_slack']:
            print(f"\n🔀 Routing to Slack Agentic MCP Server (HTTP)...")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    response = await client.post(
                        "http://localhost:8002/create_channel",
                        json={
                            "incident_id": incident.id,
                            "title": incident.title,
                            "description": incident.description,
                            "severity": incident.severity,
                            "context": {
                                "service": incident.service,
                                "error_rate": incident.error_rate,
                                "affected_users": incident.affected_users,
                                "customer_facing": incident.customer_facing,
                                "revenue_impacting": incident.revenue_impacting,
                                "region": incident.region
                            }
                        }
                    )
                    slack_result = response.json()
                    print(f"✅ Slack channel created: #{slack_result['channel_name']}")
                except Exception as e:
                    print(f"❌ Slack MCP error: {e}")
                    slack_result = {"error": str(e)}
        
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        results.append(OrchestrationResult(
            incident_id=incident.id,
            routing_decision=routing,
            jira_result=jira_result,
            slack_result=slack_result,
            orchestrator_reasoning=routing['reasoning'],
            total_time_ms=duration_ms
        ))
        
        print(f"\n✅ Completed {incident.id} in {duration_ms:.0f}ms")
    
    print(f"\n{'='*80}")
    print(f"🎉 ORCHESTRATOR COMPLETE - Processed {len(results)} incidents")
    print(f"{'='*80}\n")
    
    return results


@orchestrator_app.get("/health")
async def health_check():
    """Check health of all services"""
    
    services = {
        "orchestrator": {"status": "healthy", "port": 8000},
        "jira_mcp": {"status": "unknown", "port": 8001},
        "slack_mcp": {"status": "unknown", "port": 8002}
    }
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        # Check Jira MCP
        try:
            response = await client.get("http://localhost:8001/")
            services["jira_mcp"]["status"] = "healthy" if response.status_code == 200 else "unhealthy"
        except:
            services["jira_mcp"]["status"] = "unreachable"
        
        # Check Slack MCP
        try:
            response = await client.get("http://localhost:8002/")
            services["slack_mcp"]["status"] = "healthy" if response.status_code == 200 else "unhealthy"
        except:
            services["slack_mcp"]["status"] = "unreachable"
    
    return {
        "timestamp": datetime.now().isoformat(),
        "services": services,
        "overall": "healthy" if all(s["status"] == "healthy" for s in services.values()) else "degraded"
    }


# ============================================================================
# STARTUP SCRIPT
# ============================================================================

async def run_all_servers():
    """Run all three servers simultaneously"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                                                                          ║
    ║               LLM ORCHESTRATOR WITH AGENTIC MCP SERVERS                 ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    
    Architecture:
    
    ┌─────────────────────┐
    │  Orchestrator LLM   │  :8000  (Reads DB, Routes intelligently)
    │  (Main Controller)  │
    └──────────┬──────────┘
               │
         ┌─────┴─────┐
         │           │
         ▼           ▼
    ┌─────────┐  ┌─────────┐
    │ Jira    │  │ Slack   │
    │ Agentic │  │ Agentic │
    │ MCP     │  │ MCP     │  
    │ :8001   │  │ :8002   │  (Each has own LLM + Mock MCP)
    └────┬────┘  └────┬────┘
         │            │
         ▼            ▼
    [Mock Jira]  [Mock Slack]
    
    
    Starting servers...
    """)
    
    # Note: In production, run each server in separate processes
    print("⚠️  Run each server in a separate terminal:")
    print("")
    print("Terminal 1: uvicorn orchestrator_app:orchestrator_app --port 8000")
    print("Terminal 2: uvicorn orchestrator_app:jira_app --port 8001")
    print("Terminal 3: uvicorn orchestrator_app:slack_app --port 8002")
    print("")
    print("Then test:")
    print("curl -X POST http://localhost:8000/process")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        service = sys.argv[1]
        
        if service == "orchestrator":
            print("\n🎭 Starting ORCHESTRATOR LLM (Port 8000)...\n")
            uvicorn.run(orchestrator_app, host="0.0.0.0", port=8000)
        
        elif service == "jira":
            print("\n🎫 Starting JIRA AGENTIC MCP SERVER (Port 8001)...\n")
            uvicorn.run(jira_app, host="0.0.0.0", port=8001)
        
        elif service == "slack":
            print("\n💬 Starting SLACK AGENTIC MCP SERVER (Port 8002)...\n")
            uvicorn.run(slack_app, host="0.0.0.0", port=8002)
        
        else:
            print(f"❌ Unknown service: {service}")
            print("Usage: python script.py [orchestrator|jira|slack]")
    
    else:
        asyncio.run(run_all_servers())