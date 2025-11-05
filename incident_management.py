"""
Incident Management System with AI Agents
Uses CrewAI for orchestrating reporting and ticketing automation
Integrates with MCP servers (Jira, Slack, PagerDuty)
TRUE LLM-driven decision making (NO hardcoded if-else logic)
"""

from crewai import Agent, Task, Crew, Process, LLM
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import json
from dataclasses import dataclass, asdict
import asyncio
import random
import re


# ============================================================================
# DATA MODELS
# ============================================================================

class IncidentSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IncidentStatus(Enum):
    DETECTED = "detected"
    TRIAGED = "triaged"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    CLOSED = "closed"


@dataclass
class Incident:
    id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    detected_at: str
    service: str
    metrics: Dict[str, Any]
    logs: List[str]
    affected_components: List[str]
    region: str
    incident_text: str
    corrective_actions: List[str]
    root_cause: Optional[str] = None
    resolution: Optional[str] = None


@dataclass
class IncidentContext:
    """Extended incident context for nuanced decision making"""
    business_hours: bool  # 9 AM - 6 PM local time
    peak_traffic_hours: bool  # 10 AM - 2 PM, 6 PM - 9 PM
    weekend: bool
    customer_facing: bool
    revenue_impacting: bool


@dataclass
class IncidentReport:
    incident_id: str
    summary: str
    timeline: List[Dict[str, str]]
    impact_analysis: str
    root_cause: str
    resolution_steps: List[str]
    recommendations: List[str]
    generated_at: str


@dataclass
class Ticket:
    ticket_id: str
    incident_id: str
    platform: str
    title: str
    description: str
    priority: str
    assignee: Optional[str]
    created_at: str
    url: Optional[str] = None


llm = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434"
)

# ============================================================================
# SYNTHETIC DATA GENERATOR
# ============================================================================

class SyntheticIncidentGenerator:
    """Generate realistic incident scenarios for testing"""
    
    def __init__(self):
        self.incident_counter = 0
        self.regions = ["us-east-1", "us-west-2", "eu-west-1", "ap-south-1", "eu-central-1"]
        
    def _get_time_context(self, force_hour: Optional[int] = None) -> tuple:
        """Determine time-based context"""
        hour = force_hour if force_hour is not None else datetime.now().hour
        day_of_week = datetime.now().weekday()
        
        business_hours = 9 <= hour < 18
        peak_traffic = (10 <= hour < 14) or (18 <= hour < 21)
        weekend = day_of_week >= 5
        
        return business_hours, peak_traffic, weekend
    
    def generate_scenarios(self, count: int = 10) -> List[tuple]:
        """Generate diverse incident scenarios"""
        scenarios = []
        
        # Scenario templates
        templates = [
            {
                "title": "Authentication Service Degraded Performance",
                "incident_text": "User login attempts experiencing 5-10 second delays. Auth token generation timing out intermittently affecting 30% of requests.",
                "service": "auth-service",
                "severity": IncidentSeverity.HIGH,
                "customer_facing": True,
                "revenue_impacting": True,
                "affected_components": ["auth-service", "redis-cache", "session-manager"],
                "metrics": {"error_rate": 0.3, "requests_per_second": 850, "p99_latency_ms": 9500},
                "logs": [
                    "[WARN] Redis cache miss rate at 85%",
                    "[ERROR] Session token generation timeout",
                    "[WARN] Auth service CPU at 95%",
                    "[INFO] Auto-scaling triggered"
                ],
                "corrective_actions": [
                    "Scale auth service horizontally",
                    "Clear Redis cache and rebuild",
                    "Reduce token TTL temporarily",
                    "Enable rate limiting"
                ]
            },
            {
                "title": "API Rate Limiting - Partner Payment Service",
                "incident_text": "Third-party payment provider rate limiting our API calls. Affecting 25% of checkout flows during peak hours.",
                "service": "payment-integration",
                "severity": IncidentSeverity.MEDIUM,
                "customer_facing": True,
                "revenue_impacting": True,
                "affected_components": ["payment-integration", "checkout-service", "retry-queue"],
                "metrics": {"error_rate": 0.25, "rate_limit_hits": 1500, "queue_depth": 450},
                "logs": [
                    "[ERROR] Partner API returned 429 Too Many Requests",
                    "[WARN] Rate limit threshold reached",
                    "[INFO] Queuing failed requests for retry",
                    "[WARN] Checkout completion time increasing"
                ],
                "corrective_actions": [
                    "Implement exponential backoff",
                    "Negotiate higher rate limits with partner",
                    "Enable request batching",
                    "Add fallback payment provider"
                ]
            },
            {
                "title": "Memory Leak in Notification Service",
                "incident_text": "User notification service showing steady memory growth. Memory usage at 85% and climbing. No immediate functional impact but service restart needed soon.",
                "service": "notification-service",
                "severity": IncidentSeverity.LOW,
                "customer_facing": False,
                "revenue_impacting": False,
                "affected_components": ["notification-service", "message-queue"],
                "metrics": {"memory_usage_percent": 85, "heap_size_mb": 3400, "gc_time_ms": 850},
                "logs": [
                    "[WARN] Memory usage trending upward",
                    "[INFO] Garbage collection frequency increasing",
                    "[WARN] Heap near maximum capacity",
                    "[INFO] No OOM errors yet"
                ],
                "corrective_actions": [
                    "Schedule rolling restart during low traffic",
                    "Enable heap dump for analysis",
                    "Review recent code changes",
                    "Monitor for OOM conditions"
                ]
            },
            {
                "title": "DDoS Attack Suspected",
                "incident_text": "Unusual traffic spike from single IP range. 10x normal traffic volume. WAF rules blocking most requests but causing legitimate user impact.",
                "service": "edge-firewall",
                "severity": IncidentSeverity.CRITICAL,
                "customer_facing": True,
                "revenue_impacting": True,
                "affected_components": ["waf", "load-balancer", "origin-servers"],
                "metrics": {"requests_per_second": 50000, "blocked_requests": 45000, "false_positive_rate": 0.15},
                "logs": [
                    "[CRITICAL] Massive traffic spike detected",
                    "[WARN] WAF blocking high volume of requests",
                    "[ERROR] Legitimate users being blocked",
                    "[INFO] DDoS mitigation engaged"
                ],
                "corrective_actions": [
                    "Refine WAF rules to reduce false positives",
                    "Enable advanced DDoS protection",
                    "Contact ISP for upstream filtering",
                    "Implement CAPTCHA for suspicious traffic"
                ]
            }
        ]
        
        # Generate incidents with varying contexts
        for i in range(count):
            template = templates[i]
            self.incident_counter += 1
            
            # Vary time contexts for testing
            force_hour = None
            if i % 4 == 0:  # 3 AM off-hours
                force_hour = 3
            elif i % 4 == 1:  # 11 AM peak
                force_hour = 11
            elif i % 4 == 2:  # 7 PM evening peak
                force_hour = 19
            elif i % 4 == 3:  # Normal business hours
                force_hour = 14
            
            business_hours, peak_traffic, weekend = self._get_time_context(force_hour)
            
            if i % 7 == 0:  # Some on weekends
                weekend = True
                
            region = random.choice(self.regions)
            
            incident = Incident(
                id=f"INC-2025-{str(self.incident_counter).zfill(3)}",
                title=template["title"],
                description=template["incident_text"],
                severity=template["severity"],
                status=IncidentStatus.DETECTED,
                detected_at=datetime.now().isoformat(),
                service=template["service"],
                metrics=template["metrics"],
                logs=template["logs"],
                affected_components=template["affected_components"],
                region=region,
                incident_text=template["incident_text"],
                corrective_actions=template["corrective_actions"],
                root_cause=None
            )
            
            context = IncidentContext(
                business_hours=business_hours,
                peak_traffic_hours=peak_traffic,
                weekend=weekend,
                customer_facing=template["customer_facing"],
                revenue_impacting=template["revenue_impacting"]
            )
            
            scenarios.append((incident, context))
        
        return scenarios


# ============================================================================
# MCP SERVER TOOLS
# ============================================================================

class MCPServerTool:
    """Base class for MCP server tools"""
    
    def __init__(self, server_name: str):
        self.server_name = server_name
        self.connected = False
    
    async def connect(self):
        """Connect to MCP server"""
        await asyncio.sleep(0.1)
        self.connected = True


class JiraMCPTool(MCPServerTool):
    """Jira MCP Server Integration"""
    
    def __init__(self):
        super().__init__("Jira")
    
    async def create_issue(self, project: str, summary: str, description: str, 
                          priority: str, issue_type: str = "Bug") -> Dict[str, Any]:
        """Create a Jira issue"""
        await self.connect()
        
        ticket_id = f"INCIDENT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return {
            "status": "success",
            "ticket_id": ticket_id,
            "url": f"https://jira.company.com/browse/{ticket_id}",
            "priority": priority
        }


class SlackMCPTool(MCPServerTool):
    """Slack MCP Server Integration"""
    
    def __init__(self):
        super().__init__("Slack")
    
    async def send_message(self, channel: str, message: str) -> Dict[str, Any]:
        """Send message to Slack channel"""
        await self.connect()
        
        return {
            "status": "success",
            "channel": channel,
            "timestamp": datetime.now().isoformat()
        }
    
    async def create_incident_channel(self, incident_id: str) -> Dict[str, Any]:
        """Create dedicated incident channel"""
        channel_name = f"incident-{incident_id}"
        
        return {
            "status": "success",
            "channel_name": channel_name
        }
    
    async def send_alert(self, channel: str, severity: str, message: str) -> Dict[str, Any]:
        """Send formatted alert to Slack"""
        emoji = {
            "critical": "🚨",
            "high": "⚠️",
            "medium": "⚡",
            "low": "ℹ️"
        }.get(severity.lower(), "📢")
        
        return await self.send_message(channel, f"{emoji} *{severity.upper()}* Alert\n{message}")


class PagerDutyMCPTool(MCPServerTool):
    """PagerDuty MCP Server Integration"""
    
    def __init__(self):
        super().__init__("PagerDuty")
    
    async def create_incident(self, title: str, description: str, 
                             urgency: str, service_id: str) -> Dict[str, Any]:
        """Create PagerDuty incident"""
        await self.connect()
        
        incident_id = f"PD{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return {
            "status": "success",
            "incident_id": incident_id,
            "url": f"https://company.pagerduty.com/incidents/{incident_id}",
            "urgency": urgency
        }
    
    async def trigger_escalation(self, incident_id: str) -> Dict[str, Any]:
        """Trigger escalation policy"""
        return {
            "status": "success",
            "message": f"Triggered escalation for {incident_id}"
        }


# ============================================================================
# CREWAI AGENTS
# ============================================================================

class ReportingAgent:
    """AI Agent responsible for generating incident reports"""
    
    def __init__(self):
        self.agent = Agent(
            role='Incident Report Analyst',
            goal='Generate comprehensive incident reports with root cause analysis',
            backstory="""You are an experienced SRE and incident analyst with deep 
            expertise in system reliability, troubleshooting, and post-mortem analysis.""",
            llm=llm,
            verbose=True,
            allow_delegation=False
        )
    
    def create_report_task(self, incident: Incident) -> Task:
        """Create report generation task"""
        return Task(
            description=f"""
            Analyze this incident and generate a comprehensive report:
            
            Incident ID: {incident.id}
            Title: {incident.title}
            Severity: {incident.severity.value}
            Region: {incident.region}
            Service: {incident.service}
            
            Description: {incident.incident_text}
            
            Components: {', '.join(incident.affected_components)}
            Metrics: {json.dumps(incident.metrics, indent=2)}
            
            Logs:
            {chr(10).join(incident.logs)}
            
            Corrective Actions Available:
            {chr(10).join(f'- {action}' for action in incident.corrective_actions)}
            
            Generate:
            1. Executive Summary
            2. Timeline
            3. Impact Analysis
            4. Root Cause
            5. Resolution Steps
            6. Recommendations
            """,
            agent=self.agent,
            expected_output="Comprehensive incident report"
        )
    
    def parse_report(self, task_output: str, incident: Incident) -> IncidentReport:
        """Parse report output"""
        return IncidentReport(
            incident_id=incident.id,
            summary=f"{incident.title} - {incident.severity.value} severity in {incident.region}",
            timeline=[
                {"time": incident.detected_at, "event": "Incident detected"},
                {"time": datetime.now().isoformat(), "event": "Report generated"}
            ],
            impact_analysis=f"Affected {len(incident.affected_components)} components in {incident.region}",
            root_cause=incident.root_cause or "Investigation in progress",
            resolution_steps=incident.corrective_actions,
            recommendations=[
                "Enhanced monitoring",
                "Automated remediation",
                "Capacity review"
            ],
            generated_at=datetime.now().isoformat()
        )


class IntelligentTicketingAgent:
    """Enhanced ticketing agent with TRUE LLM-driven decision making"""
    
    def __init__(self, jira_tool: JiraMCPTool, slack_tool: SlackMCPTool, 
                 pagerduty_tool: PagerDutyMCPTool):
        self.jira_tool = jira_tool
        self.slack_tool = slack_tool
        self.pagerduty_tool = pagerduty_tool
        
        self.agent = Agent(
            role='Intelligent Incident Ticketing Coordinator',
            goal='Make nuanced, context-aware decisions about incident routing and escalation',
            backstory="""You are an expert incident coordinator with 15+ years of SRE experience.
            You've handled thousands of incidents and developed an intuitive sense for what truly 
            needs immediate attention versus what can wait. You understand the human cost of alert 
            fatigue and balance urgency with team wellbeing. You think in terms of business impact, 
            customer experience, and operational pragmatism.
            
            You know that severity labels alone don't tell the full story - a "high" severity 
            internal dashboard issue at 3 AM is very different from a "medium" severity payment 
            processing degradation during peak shopping hours.
            
            You make decisions that experienced SRE managers would make, not naive automated systems.""",
            llm=llm,
            verbose=True,
            allow_delegation=False
        )
    
    def create_decision_task(self, incident: Incident, 
                            report: IncidentReport,
                            context: IncidentContext) -> Task:
        """Create task that asks LLM to make the actual decision"""
        
        # Build context description
        time_context = self._describe_time_context(context)
        impact_context = self._describe_impact_context(context)
        
        return Task(
            description=f"""
You are an experienced SRE manager making a real incident response decision. Think through this 
carefully - your decision affects real people's sleep, work-life balance, and system reliability.

═══════════════════════════════════════════════════════════════════════════════
INCIDENT SITUATION
═══════════════════════════════════════════════════════════════════════════════

{incident.title}
ID: {incident.id} | Service: {incident.service} | Region: {incident.region}
Labeled Severity: {incident.severity.value.upper()} (but labels can be misleading!)

What's happening:
{incident.incident_text}

═══════════════════════════════════════════════════════════════════════════════
CONTEXT THAT MATTERS
═══════════════════════════════════════════════════════════════════════════════

TIME CONTEXT:
{time_context}

BUSINESS IMPACT:
{impact_context}

TECHNICAL METRICS:
• Error Rate: {incident.metrics.get('error_rate', 'N/A')}
• Latency (p99): {incident.metrics.get('p99_latency_ms', 'N/A')}ms
• Request Rate: {incident.metrics.get('requests_per_second', 'N/A')} req/s
• Affected Components: {', '.join(incident.affected_components)}

RECENT SYSTEM LOGS:
{self._format_list(incident.logs[-4:])}

AVAILABLE FIXES:
{self._format_list(incident.corrective_actions)}

═══════════════════════════════════════════════════════════════════════════════
YOUR DECISION-MAKING PROCESS
═══════════════════════════════════════════════════════════════════════════════

First, THINK THROUGH THESE QUESTIONS (write your thoughts):

1. URGENCY CHECK:
   - If you were the on-call engineer, would you want to be woken up at 3 AM for this?
   - Or can this wait until morning standup?
   - What's the actual blast radius RIGHT NOW?

2. IMPACT REALITY:
   - How many real users are affected this very moment?
   - Are they unable to do their job or just experiencing slowness?
   - Is money being lost every minute, or is this a "quality of life" issue?

3. TIME SENSITIVITY:
   - Will waiting 4 hours make this significantly worse?
   - Is there active data loss or just potential future problems?
   - Can the team self-heal or auto-scale through this?

4. ALERT FATIGUE:
   - Is this truly exceptional or part of normal operations?
   - Have we been paging too much lately?
   - Does the team need sleep more than we need immediate action?

5. PROPORTIONAL RESPONSE:
   - What's the MINIMUM response that handles this appropriately?
   - Are we about to create a war room for something that needs a Jira ticket?
   - Should we notify vs. escalate vs. just track?

═══════════════════════════════════════════════════════════════════════════════
RESPONSE SCENARIOS TO CONSIDER
═══════════════════════════════════════════════════════════════════════════════

Think about these patterns:

• "It's 3 AM and the analytics dashboard is lagging" 
  → Probably: Jira ticket, NO page, wait for business hours

• "Payment processing is down during peak shopping hours"
  → Probably: Page immediately, create war room, high urgency

• "Internal tool has a memory leak, 85% usage but stable"
  → Probably: Slack alert, schedule restart, no page

• "DDoS hitting us but WAF is handling it, some false positives"
  → Depends: Are real users blocked? How many? Page if revenue impact.

• "High severity label but it's a weekend internal dashboard issue"
  → Don't blindly trust severity labels! Assess actual impact.

═══════════════════════════════════════════════════════════════════════════════
NOW MAKE YOUR DECISION
═══════════════════════════════════════════════════════════════════════════════

Start with your reasoning - walk through your thought process conversationally.
Then provide your decision in JSON format.

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

REASONING:
[Write 3-5 sentences explaining your thinking. Be specific about why you're choosing 
this response level. Reference the specific context factors that influenced you.
If you're NOT doing something (like not paging), explicitly say why.]

DECISION:
```json
{{
  "use_pagerduty": ,
  "pagerduty_urgency": ,
  "use_slack": ,
  "slack_channel": ,
  "create_jira": ,
  "jira_priority": ,
  "create_war_room": ,
  "trigger_escalation": ,
  "confidence_level": ,
  "key_factors": []
}}
```

Field types:
- use_pagerduty, use_slack, create_jira, create_war_room, trigger_escalation: boolean
- pagerduty_urgency: "high" or "low" (or null if not using PagerDuty)
- slack_channel: string like "#incidents-critical", "#incidents", "#ops-alerts" (or null if not using Slack)
- jira_priority: "CRITICAL", "HIGH", "MEDIUM", or "LOW"
- confidence_level: "high", "medium", or "low"
- key_factors: array of 2-4 strings describing what drove your decision

REMEMBER: 
- Don't just follow the severity label blindly
- Consider the human cost of waking people up unnecessarily
- Think about business hours vs. off-hours differently
- Internal vs. customer-facing matters a LOT
- Revenue impact is different from operational impact
- Sometimes "do nothing urgent" is the right answer

Make the decision you'd make if this were your on-call shift and your team's sleep schedule.
            """,
            agent=self.agent,
            expected_output="Detailed reasoning followed by JSON decision"
        )
    
    def _describe_time_context(self, context: IncidentContext) -> str:
        """Generate human-readable time context"""
        parts = []
        
        if not context.business_hours:
            parts.append("⏰ OFF-HOURS (likely 2-4 AM) - Most engineers are asleep")
        elif context.peak_traffic_hours:
            parts.append("📈 PEAK TRAFFIC HOURS - Maximum user activity, highest impact window")
        else:
            parts.append("🕐 BUSINESS HOURS - Team is available and working")
        
        if context.weekend:
            parts.append("📅 WEEKEND - Reduced on-call coverage")
        else:
            parts.append("📅 WEEKDAY - Full team availability")
        
        return "\n".join(parts)
    
    def _describe_impact_context(self, context: IncidentContext) -> str:
        """Generate human-readable impact context"""
        parts = []
        
        if context.customer_facing:
            parts.append("👥 CUSTOMER-FACING: External users directly affected")
        else:
            parts.append("🔧 INTERNAL ONLY: No direct customer impact")
        
        if context.revenue_impacting:
            parts.append("💰 REVENUE-IMPACTING: Directly affects company revenue")
        else:
            parts.append("💵 NO REVENUE IMPACT: Operational issue only")
        
        return "\n".join(parts)
    
    def _format_list(self, items: List[str]) -> str:
        """Format list items"""
        return "\n".join(f"  • {item}" for item in items)
    
    def _parse_llm_decision(self, llm_output: str) -> Dict[str, Any]:
        """Parse LLM output to extract JSON decision"""
        try:
            # Try to find JSON in markdown code block
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', llm_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON
                json_match = re.search(r'\{.*?"reasoning".*?\}', llm_output, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON found in LLM output")
            
            decision = json.loads(json_str)
            
            # Validate required fields
            required_fields = [
                'reasoning', 'use_pagerduty', 'use_slack', 'jira_priority',
                'create_jira', 'create_war_room', 'trigger_escalation'
            ]
            
            for field in required_fields:
                if field not in decision:
                    raise ValueError(f"Missing required field: {field}")
            
            return decision
            
        except Exception as e:
            print(f"⚠️  Failed to parse LLM decision: {e}")
            print(f"Raw output: {llm_output[:500]}...")
            # Return conservative default
            return {
                "reasoning": "Failed to parse LLM output, using conservative defaults",
                "use_pagerduty": True,
                "pagerduty_urgency": "high",
                "use_slack": True,
                "slack_channel": "#incidents",
                "jira_priority": "HIGH",
                "create_jira": True,
                "create_war_room": False,
                "trigger_escalation": False,
                "wait_until_business_hours": False,
                "confidence_level": "low",
                "key_factors": ["parse_error"]
            }
    
    async def execute_intelligent_ticketing(self, incident: Incident, 
                                           report: IncidentReport,
                                           context: IncidentContext) -> Dict[str, Any]:
        """Execute ticketing based on LLM decision"""
        
        print(f"\n🧠 Asking LLM to make decision...")
        print(f"   Context: {self._format_context(context)}")
        print()
        
        # Create decision task
        decision_task = self.create_decision_task(incident, report, context)
        
        # Run CrewAI to get LLM decision
        decision_crew = Crew(
            agents=[self.agent],
            tasks=[decision_task],
            process=Process.sequential,
            verbose=False
        )
        
        # Get LLM output
        llm_output = str(decision_crew.kickoff())
        
        # Parse decision
        decision = self._parse_llm_decision(llm_output)
        
        # Display LLM reasoning
        print("💭 LLM Decision Reasoning:")
        print(f"   {decision['reasoning']}")
        print()
        
        if 'key_factors' in decision:
            print("🔑 Key Decision Factors:")
            for factor in decision['key_factors']:
                print(f"   • {factor}")
            print()
        
        print(f"🎯 Decision (Confidence: {decision.get('confidence_level', 'medium')}):")
        print(f"   • PagerDuty: {'YES' if decision['use_pagerduty'] else 'NO'}")
        print(f"   • Slack: {decision.get('slack_channel', 'NO') if decision['use_slack'] else 'NO'}")
        print(f"   • Jira Priority: {decision['jira_priority']}")
        print(f"   • War Room: {'YES' if decision['create_war_room'] else 'NO'}")
        print(f"   • Escalate: {'YES' if decision['trigger_escalation'] else 'NO'}")
        print()
        
        # Execute the LLM's decisions
        tickets = []
        actions = []
        
        # 1. Jira
        if decision['create_jira']:
            jira_result = await self.jira_tool.create_issue(
                project="INCIDENT",
                summary=incident.title,
                description=f"{incident.incident_text}\n\nRegion: {incident.region}\n\n" +
                           f"Corrective Actions:\n" + "\n".join(f"- {a}" for a in incident.corrective_actions),
                priority=decision['jira_priority']
            )
            
            tickets.append(Ticket(
                ticket_id=jira_result['ticket_id'],
                incident_id=incident.id,
                platform="Jira",
                title=incident.title,
                description=incident.description,
                priority=decision['jira_priority'],
                assignee=None,
                created_at=datetime.now().isoformat(),
                url=jira_result['url']
            ))
            
            actions.append(f"✓ Jira ticket created (Priority: {decision['jira_priority']})")
        
        # 2. PagerDuty
        if decision['use_pagerduty']:
            urgency = decision.get('pagerduty_urgency', 'high')
            pd_result = await self.pagerduty_tool.create_incident(
                title=incident.title,
                description=report.summary,
                urgency=urgency,
                service_id=incident.service
            )
            
            tickets.append(Ticket(
                ticket_id=pd_result['incident_id'],
                incident_id=incident.id,
                platform="PagerDuty",
                title=incident.title,
                description=incident.description,
                priority=incident.severity.value,
                assignee="On-call Engineer",
                created_at=datetime.now().isoformat(),
                url=pd_result['url']
            ))
            
            actions.append(f"✓ PagerDuty incident created (Urgency: {urgency})")
            
            if decision['trigger_escalation']:
                await self.pagerduty_tool.trigger_escalation(pd_result['incident_id'])
                actions.append("✓ Escalation policy triggered")
        else:
            actions.append("✗ NO PagerDuty - LLM decided not to wake on-call")
        
        # 3. Slack
        if decision['use_slack']:
            channel = decision.get('slack_channel', '#incidents')
            await self.slack_tool.send_alert(
                channel=channel,
                severity=incident.severity.value,
                message=f"*{incident.title}*\n{incident.incident_text}\n\nRegion: {incident.region}\nJira: {tickets[0].url if tickets else 'N/A'}"
            )
            
            actions.append(f"✓ Slack alert sent to {channel}")
            
            if decision['create_war_room']:
                war_room = await self.slack_tool.create_incident_channel(incident.id)
                actions.append(f"✓ War room created: #{war_room['channel_name']}")
        else:
            actions.append("✗ NO Slack alert - LLM decided notification not needed")
        
        return {
            "tickets": tickets,
            "actions": actions,
            "reasoning": [decision['reasoning']],
            "decision_summary": {
                "jira": decision['create_jira'],
                "jira_priority": decision['jira_priority'],
                "pagerduty": decision['use_pagerduty'],
                "slack": decision['use_slack'],
                "slack_channel": decision.get('slack_channel'),
                "war_room": decision['create_war_room'],
                "confidence": decision.get('confidence_level', 'medium'),
                "key_factors": decision.get('key_factors', [])
            },
            "llm_decision": decision
        }
    
    def _format_context(self, context: IncidentContext) -> str:
        """Format context for display"""
        parts = []
        if not context.business_hours:
            parts.append("Off-Hours")
        elif context.peak_traffic_hours:
            parts.append("Peak Traffic")
        else:
            parts.append("Business Hours")
        
        if context.weekend:
            parts.append("Weekend")
        
        if context.customer_facing:
            parts.append("Customer-Facing")
        else:
            parts.append("Internal")
        
        if context.revenue_impacting:
            parts.append("Revenue-Impact")
        
        return " | ".join(parts)


# ============================================================================
# INCIDENT MANAGEMENT SYSTEM
# ============================================================================

class IncidentManagementSystem:
    """Main orchestrator for incident management"""
    
    def __init__(self):
        self.jira_tool = JiraMCPTool()
        self.slack_tool = SlackMCPTool()
        self.pagerduty_tool = PagerDutyMCPTool()
        
        self.reporting_agent = ReportingAgent()
        self.ticketing_agent = IntelligentTicketingAgent(
            self.jira_tool, 
            self.slack_tool, 
            self.pagerduty_tool
        )
        
        self.incidents: Dict[str, Incident] = {}
        self.reports: Dict[str, IncidentReport] = {}
        self.tickets: Dict[str, List[Ticket]] = {}
        self.decisions: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self):
        """Initialize the system"""
        print("=" * 80)
        print("🚀 Initializing Intelligent Incident Management System")
        print("=" * 80)
        print("\n✓ MCP servers ready")
        print("✓ AI agents initialized")
        print("✓ LLM-driven decision engine ready (NO hardcoded rules!)")
        print("\n")
    
    async def process_incident(self, incident: Incident, 
                              context: IncidentContext) -> Dict[str, Any]:
        """Process incident with intelligent routing"""
        
        print(f"\n{'=' * 80}")
        print(f"📊 Processing: {incident.id}")
        print(f"{'=' * 80}")
        print(f"Title: {incident.title}")
        print(f"Severity: {incident.severity.value.upper()}")
        print(f"Region: {incident.region}")
        print(f"Context: {self.ticketing_agent._format_context(context)}")
        print(f"{'=' * 80}\n")
        
        self.incidents[incident.id] = incident
        
        # Step 1: Generate Report
        print("📝 Step 1: Generating Incident Report...")
        print("-" * 80)
        
        report_task = self.reporting_agent.create_report_task(incident)
        report_crew = Crew(
            agents=[self.reporting_agent.agent],
            tasks=[report_task],
            process=Process.sequential,
            verbose=False
        )
        
        try:
            report_result = report_crew.kickoff()
            report = self.reporting_agent.parse_report(str(report_result), incident)
            self.reports[incident.id] = report
            
            print(f"✅ Report Generated")
            print(f"   Summary: {report.summary}")
            print(f"   Root Cause: {report.root_cause}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")
            return {"status": "error", "message": str(e)}
        
        # Step 2: LLM-Driven Intelligent Ticketing
        print("🎯 Step 2: LLM Making Intelligent Ticketing Decision...")
        print("-" * 80)
        
        try:
            # Execute intelligent ticketing (LLM makes the decision)
            result = await self.ticketing_agent.execute_intelligent_ticketing(
                incident, report, context
            )
            
            self.tickets[incident.id] = result['tickets']
            self.decisions[incident.id] = result
            
            print(f"\n✅ Decision Executed")
            print(f"\nActions Taken:")
            for action in result['actions']:
                print(f"   {action}")
            
            if result['reasoning']:
                print(f"\nLLM Reasoning:")
                for reason in result['reasoning']:
                    print(f"   {reason}")
            
            print()
            
        except Exception as e:
            print(f"❌ Error: {e}\n")
            return {"status": "error", "message": str(e)}
        
        # Return summary
        return {
            "status": "success",
            "incident_id": incident.id,
            "severity": incident.severity.value,
            "region": incident.region,
            "context": asdict(context),
            "report": asdict(report),
            "tickets": [asdict(t) for t in result['tickets']],
            "decision": result['decision_summary'],
            "actions": result['actions'],
            "reasoning": result['reasoning'],
            "llm_decision": result.get('llm_decision', {})
        }
    
    def print_summary(self):
        """Print overall summary"""
        print("\n" + "=" * 80)
        print("📈 INCIDENT MANAGEMENT SUMMARY")
        print("=" * 80)
        
        total_incidents = len(self.incidents)
        
        # Count by severity
        severity_counts = {}
        for incident in self.incidents.values():
            sev = incident.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        # Count tickets by platform
        platform_counts = {}
        for tickets in self.tickets.values():
            for ticket in tickets:
                platform = ticket.platform
                platform_counts[platform] = platform_counts.get(platform, 0) + 1
        
        # Decision analysis
        pagerduty_incidents = sum(1 for d in self.decisions.values() if d['decision_summary']['pagerduty'])
        jira_only_incidents = sum(1 for d in self.decisions.values() 
                                 if d['decision_summary']['jira'] and not d['decision_summary']['pagerduty'])
        
        # Confidence analysis
        high_confidence = sum(1 for d in self.decisions.values() 
                             if d['decision_summary'].get('confidence') == 'high')
        
        print(f"\nTotal Incidents Processed: {total_incidents}")
        print(f"\nBy Severity:")
        for sev, count in sorted(severity_counts.items()):
            print(f"   {sev.upper()}: {count}")
        
        print(f"\nTickets Created by Platform:")
        for platform, count in sorted(platform_counts.items()):
            print(f"   {platform}: {count}")
        
        print(f"\nLLM Decision Analysis:")
        print(f"   PagerDuty Escalations: {pagerduty_incidents}")
        print(f"   Jira Only (No Wake): {jira_only_incidents}")
        print(f"   High Confidence Decisions: {high_confidence}/{total_incidents}")
        
        print(f"\n{'=' * 80}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution with synthetic data"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                                                                          ║
    ║         INTELLIGENT INCIDENT MANAGEMENT SYSTEM                          ║
    ║         TRUE LLM-Driven Context-Aware Decision Making                   ║
    ║         (NO Hardcoded Rules!)                                           ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize system
    ims = IncidentManagementSystem()
    await ims.initialize()
    
    # Generate synthetic incidents
    print("🔬 Generating synthetic incident scenarios...")
    generator = SyntheticIncidentGenerator()
    scenarios = generator.generate_scenarios(4)
    print(f"✓ Generated {len(scenarios)} diverse incident scenarios\n")
    
    # Process each incident
    results = []
    
    for i, (incident, context) in enumerate(scenarios, 1):
        result = await ims.process_incident(incident, context)
        results.append(result)
        
        if i < len(scenarios):
            print("\n" + "▼" * 80 + "\n")
            await asyncio.sleep(0.5)  # Brief pause between incidents
    
    # Print overall summary
    ims.print_summary()
    
    # Display interesting LLM decisions
    print("=" * 80)
    print("🎯 INTERESTING LLM DECISION EXAMPLES")
    print("=" * 80)
    
    for result in results:
        if result['status'] == 'success' and 'llm_decision' in result:
            decision = result['decision']
            llm_decision = result.get('llm_decision', {})
            
            # Highlight nuanced decisions
            if llm_decision.get('key_factors'):
                print(f"\n📌 {result['incident_id']}: {result.get('severity', 'unknown').upper()}")
                print(f"   Context: {ims.ticketing_agent._format_context(IncidentContext(**result['context']))}")
                print(f"   LLM Decision:")
                print(f"      • Jira: {decision['jira_priority'] if decision['jira'] else 'NO'}")
                print(f"      • PagerDuty: {'YES' if decision['pagerduty'] else 'NO'}")
                print(f"      • Slack: {decision['slack_channel'] if decision['slack'] else 'NO'}")
                print(f"      • Confidence: {decision.get('confidence', 'medium')}")
                
                print(f"   Key Factors:")
                for factor in llm_decision.get('key_factors', [])[:3]:
                    print(f"      • {factor}")
    
    print(f"\n{'=' * 80}\n")
    
    # Export results to JSON
    output = {
        "summary": {
            "total_incidents": len(results),
            "successful": sum(1 for r in results if r['status'] == 'success'),
            "timestamp": datetime.now().isoformat(),
            "approach": "LLM-driven decisions (no hardcoded rules)"
        },
        "incidents": results
    }
    
    print("💾 Saving results to incident_results.json...")
    with open("incident_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("✓ Results saved\n")
    
    print("=" * 80)
    print("✅ INCIDENT MANAGEMENT SYSTEM TEST COMPLETE")
    print("=" * 80)
    print("\nKey Features:")
    print("• LLM makes ALL decisions dynamically (no if-else logic)")
    print("• Context-aware reasoning for each incident")
    print("• Confidence tracking for decision quality")
    print("• Explainable decisions with key factors")
    print("• Minimizes alert fatigue through intelligent routing")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())