"""
Incident Management System with AI Agents
Uses CrewAI for orchestrating reporting and ticketing automation
Integrates with MCP servers (Jira, Slack, PagerDuty)
TRUE LLM-driven decision making (NO hardcoded if-else logic)
WITH PERSISTENT REPORT STORAGE
"""

from crewai import Agent, Task, Crew, Process, LLM
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import json
from dataclasses import dataclass, asdict
from pathlib import Path
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
    business_hours: bool
    peak_traffic_hours: bool
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
    llm_analysis: Optional[str] = None  # Store raw LLM output


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
# REPORT STORAGE MANAGER
# ============================================================================

class ReportStorageManager:
    """Manages persistent storage of incident reports"""
    
    def __init__(self, base_dir: str = "incident_reports"):
        self.base_dir = Path(base_dir)
        self.reports_dir = self.base_dir / "reports"
        self.markdown_dir = self.base_dir / "markdown"
        self.json_dir = self.base_dir / "json"
        
        # Create directories
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories"""
        for directory in [self.reports_dir, self.markdown_dir, self.json_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Report storage initialized at: {self.base_dir.absolute()}")
    
    def save_report_json(self, report: IncidentReport, incident: Incident) -> Path:
        """Save report as JSON file"""
        filename = f"{report.incident_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.json_dir / filename
        
        report_data = {
            "report": asdict(report),
            "incident": asdict(incident),
            "saved_at": datetime.now().isoformat()
        }
        
        # Convert enums to strings
        report_data["incident"]["severity"] = incident.severity.value
        report_data["incident"]["status"] = incident.status.value
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        return filepath
    
    def save_report_markdown(self, report: IncidentReport, incident: Incident, 
                           context: IncidentContext) -> Path:
        """Save report as Markdown file"""
        filename = f"{report.incident_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = self.markdown_dir / filename
        
        markdown_content = self._generate_markdown(report, incident, context)
        
        with open(filepath, 'w') as f:
            f.write(markdown_content)
        
        return filepath
    
    def _generate_markdown(self, report: IncidentReport, incident: Incident, 
                          context: IncidentContext) -> str:
        """Generate markdown formatted report"""
        md = []
        
        # Header
        md.append(f"# Incident Report: {incident.title}")
        md.append(f"\n**Incident ID:** `{incident.id}`")
        md.append(f"**Generated:** {report.generated_at}")
        md.append(f"**Severity:** {incident.severity.value.upper()}")
        md.append(f"**Region:** {incident.region}")
        md.append(f"**Service:** {incident.service}")
        md.append("\n---\n")
        
        # Context
        md.append("## Context")
        md.append(f"- **Business Hours:** {'Yes' if context.business_hours else 'No'}")
        md.append(f"- **Peak Traffic:** {'Yes' if context.peak_traffic_hours else 'No'}")
        md.append(f"- **Weekend:** {'Yes' if context.weekend else 'No'}")
        md.append(f"- **Customer Facing:** {'Yes' if context.customer_facing else 'No'}")
        md.append(f"- **Revenue Impacting:** {'Yes' if context.revenue_impacting else 'No'}")
        md.append("\n")
        
        # Executive Summary
        md.append("## Executive Summary")
        md.append(f"{report.summary}\n")
        
        # Description
        md.append("## Incident Description")
        md.append(f"{incident.incident_text}\n")
        
        # Timeline
        md.append("## Timeline")
        for event in report.timeline:
            md.append(f"- **{event['time']}**: {event['event']}")
        md.append("\n")
        
        # Impact Analysis
        md.append("## Impact Analysis")
        md.append(f"{report.impact_analysis}\n")
        
        # Technical Metrics
        md.append("## Technical Metrics")
        for key, value in incident.metrics.items():
            md.append(f"- **{key.replace('_', ' ').title()}:** {value}")
        md.append("\n")
        
        # Affected Components
        md.append("## Affected Components")
        for component in incident.affected_components:
            md.append(f"- {component}")
        md.append("\n")
        
        # Root Cause
        md.append("## Root Cause Analysis")
        md.append(f"{report.root_cause}\n")
        
        # Resolution Steps
        md.append("## Resolution Steps")
        for i, step in enumerate(report.resolution_steps, 1):
            md.append(f"{i}. {step}")
        md.append("\n")
        
        # Recent Logs
        md.append("## Recent Logs")
        md.append("```")
        for log in incident.logs[-5:]:
            md.append(log)
        md.append("```\n")
        
        # Recommendations
        md.append("## Recommendations")
        for rec in report.recommendations:
            md.append(f"- {rec}")
        md.append("\n")
        
        # LLM Analysis (if available)
        if report.llm_analysis:
            md.append("## AI Analysis")
            md.append("```")
            md.append(report.llm_analysis)
            md.append("```\n")
        
        md.append("---\n")
        md.append(f"*Report generated by Intelligent Incident Management System*")
        
        return "\n".join(md)
    
    def get_all_reports(self) -> List[Dict[str, Any]]:
        """Load all stored reports"""
        reports = []
        
        for json_file in self.json_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                reports.append(json.load(f))
        
        return reports
    
    def export_summary(self, output_file: str = "reports_summary.json"):
        """Export summary of all reports"""
        all_reports = self.get_all_reports()
        
        summary = {
            "total_reports": len(all_reports),
            "generated_at": datetime.now().isoformat(),
            "reports": all_reports
        }
        
        filepath = self.base_dir / output_file
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📊 Summary exported to: {filepath}")
        return filepath


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
            if i % 4 == 0:
                force_hour = 3
            elif i % 4 == 1:
                force_hour = 11
            elif i % 4 == 2:
                force_hour = 19
            elif i % 4 == 3:
                force_hour = 14
            
            business_hours, peak_traffic, weekend = self._get_time_context(force_hour)
            
            if i % 7 == 0:
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
# MCP SERVER TOOLS (keeping original code)
# ============================================================================

class MCPServerTool:
    def __init__(self, server_name: str):
        self.server_name = server_name
        self.connected = False
    
    async def connect(self):
        await asyncio.sleep(0.1)
        self.connected = True


class JiraMCPTool(MCPServerTool):
    def __init__(self):
        super().__init__("Jira")
    
    async def create_issue(self, project: str, summary: str, description: str, 
                          priority: str, issue_type: str = "Bug") -> Dict[str, Any]:
        await self.connect()
        ticket_id = f"INCIDENT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        return {
            "status": "success",
            "ticket_id": ticket_id,
            "url": f"https://jira.company.com/browse/{ticket_id}",
            "priority": priority
        }


class SlackMCPTool(MCPServerTool):
    def __init__(self):
        super().__init__("Slack")
    
    async def send_message(self, channel: str, message: str) -> Dict[str, Any]:
        await self.connect()
        return {
            "status": "success",
            "channel": channel,
            "timestamp": datetime.now().isoformat()
        }
    
    async def create_incident_channel(self, incident_id: str) -> Dict[str, Any]:
        channel_name = f"incident-{incident_id}"
        return {
            "status": "success",
            "channel_name": channel_name
        }
    
    async def send_alert(self, channel: str, severity: str, message: str) -> Dict[str, Any]:
        emoji = {
            "critical": "🚨",
            "high": "⚠️",
            "medium": "⚡",
            "low": "ℹ️"
        }.get(severity.lower(), "📢")
        return await self.send_message(channel, f"{emoji} *{severity.upper()}* Alert\n{message}")


class PagerDutyMCPTool(MCPServerTool):
    def __init__(self):
        super().__init__("PagerDuty")
    
    async def create_incident(self, title: str, description: str, 
                             urgency: str, service_id: str) -> Dict[str, Any]:
        await self.connect()
        incident_id = f"PD{datetime.now().strftime('%Y%m%d%H%M%S')}"
        return {
            "status": "success",
            "incident_id": incident_id,
            "url": f"https://company.pagerduty.com/incidents/{incident_id}",
            "urgency": urgency
        }
    
    async def trigger_escalation(self, incident_id: str) -> Dict[str, Any]:
        return {
            "status": "success",
            "message": f"Triggered escalation for {incident_id}"
        }


# ============================================================================
# CREWAI AGENTS (keeping original ReportingAgent)
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
        """Parse report output and store LLM analysis"""
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
            generated_at=datetime.now().isoformat(),
            llm_analysis=task_output  # Store raw LLM output
        )


# Note: IntelligentTicketingAgent class remains the same as in your original code
# (I'm omitting it here for brevity, but you would keep your updated version with the improved prompt)


# ============================================================================
# INCIDENT MANAGEMENT SYSTEM (UPDATED WITH STORAGE)
# ============================================================================

class IncidentManagementSystem:
    """Main orchestrator for incident management with report storage"""
    
    def __init__(self, storage_dir: str = "incident_reports"):
        self.jira_tool = JiraMCPTool()
        self.slack_tool = SlackMCPTool()
        self.pagerduty_tool = PagerDutyMCPTool()
        
        self.reporting_agent = ReportingAgent()
        # Include your IntelligentTicketingAgent initialization here
        
        self.incidents: Dict[str, Incident] = {}
        self.reports: Dict[str, IncidentReport] = {}
        self.tickets: Dict[str, List[Ticket]] = {}
        self.decisions: Dict[str, Dict[str, Any]] = {}
        self.contexts: Dict[str, IncidentContext] = {}
        
        # Initialize storage manager
        self.storage = ReportStorageManager(storage_dir)
    
    async def initialize(self):
        """Initialize the system"""
        print("=" * 80)
        print("🚀 Initializing Intelligent Incident Management System")
        print("=" * 80)
        print("\n✓ MCP servers ready")
        print("✓ AI agents initialized")
        print("✓ LLM-driven decision engine ready")
        print("✓ Report storage configured")
        print("\n")
    
    async def process_incident(self, incident: Incident, 
                              context: IncidentContext) -> Dict[str, Any]:
        """Process incident with intelligent routing and report storage"""
        
        print(f"\n{'=' * 80}")
        print(f"📊 Processing: {incident.id}")
        print(f"{'=' * 80}")
        print(f"Title: {incident.title}")
        print(f"Severity: {incident.severity.value.upper()}")
        print(f"Region: {incident.region}")
        print(f"{'=' * 80}\n")
        
        self.incidents[incident.id] = incident
        self.contexts[incident.id] = context
        
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
            
            # ===== SAVE REPORT TO STORAGE =====
            json_path = self.storage.save_report_json(report, incident)
            md_path = self.storage.save_report_markdown(report, incident, context)
            
            print(f"✅ Report Generated")
            print(f"   Summary: {report.summary}")
            print(f"   Root Cause: {report.root_cause}")
            print(f"   💾 Saved to JSON: {json_path.name}")
            print(f"   💾 Saved to Markdown: {md_path.name}\n")
            
        except Exception as e:
            print(f"❌ Error: {e}\n")
            return {"status": "error", "message": str(e)}
        
        # Step 2: Continue with ticketing logic...
        # (Include your existing ticketing code here)
        
        return {
            "status": "success",
            "incident_id": incident.id,
            "severity": incident.severity.value,
            "region": incident.region,
            "report_paths": {
                "json": str(json_path),
                "markdown": str(md_path)
            }
        }
    
    def export_all_reports(self):
        """Export all reports to summary file"""
        print("\n" + "=" * 80)
        print("💾 EXPORTING ALL REPORTS")
        print("=" * 80)
        
        self.storage.export_summary()
        
        print(f"\n📊 Total Reports: {len(self.reports)}")
        print(f"📁 Storage Location: {self.storage.base_dir.absolute()}")
        print(f"   • JSON Reports: {len(list(self.storage.json_dir.glob('*.json')))}")
        print(f"   • Markdown Reports: {len(list(self.storage.markdown_dir.glob('*.md')))}")
        print("=" * 80 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution with report storage"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                                                                          ║
    ║         INTELLIGENT INCIDENT MANAGEMENT SYSTEM                          ║
    ║         With Persistent Report Storage                                  ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize system
    ims = IncidentManagementSystem(storage_dir="incident_reports")
    await ims.initialize()
    
    # Generate synthetic incidents
    print("🔬 Generating synthetic incident scenarios...")
    generator = SyntheticIncidentGenerator()
    scenarios = generator.generate_scenarios(count=4)
    print(f"✓ Generated {len(scenarios)} diverse incident scenarios\n")
    
    # Process each incident
    results = []
    
    for i, (incident, context) in enumerate(scenarios, 1):
        result = await ims.process_incident(incident, context)
        results.append(result)
        
        if i < len(scenarios):
            print("\n" + "▼" * 80 + "\n")
            await asyncio.sleep(0.5)
    
    # Export all reports
    ims.export_all_reports()
    
    print("=" * 80)
    print("✅ INCIDENT MANAGEMENT SYSTEM TEST COMPLETE")
    print("=" * 80)
    print("\nStored Reports:")
    print(f"• JSON format in: {ims.storage.json_dir}")
    print(f"• Markdown format in: {ims.storage.markdown_dir}")
    print(f"• Summary at: {ims.storage.base_dir / 'reports_summary.json'}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())