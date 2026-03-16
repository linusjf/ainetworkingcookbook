#!/usr/bin/env python
"""
Directedprompt.

######################################################################
# @author      : Linus Fernandes (linusfernandes at gmail dot com)
# @file        : directedprompt
# @created     : Monday Mar 16, 2026 15:12:29 IST
# @description :
# -*- coding: utf-8 -*-'
######################################################################
"""
import argparse
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_bgp_config(harden_security: bool = False) -> str:
    """Generate BGP configuration prompt with optional security hardening."""
    base_prompt = """
You are a senior network engineer specializing in BGP routing
protocols.
I need you to generate a BGP configuration for a Cisco ASR9000
router that will:
1. Establish EBGP peering with two upstream ISPs (AS 65001 and AS
65002)
2. Implement route filtering to accept only default routes from
upstreams
3. Set up load balancing between the two upstreams
4. Include basic security hardening for BGP sessions
My router's AS number is 65100, and the peer IPs are:
- ISP1 (AS 65001): 192.168.1.1
- ISP2 (AS 65002): 192.168.2.1
Provide the configuration in IOS-XR format with explanatory comments.
"""
    
    if harden_security:
        enhanced_prompt = base_prompt + """
Additionally, implement enhanced BGP security hardening including:
- BGP TTL security (GTSM)
- Maximum prefix limits with restart timer
- MD5 authentication for BGP sessions
- Route flap damping
- BGP route refresh capability
- Strict inbound/outbound route filtering
- Log neighbor changes
- Implement BGP prefix validation using RPKI (if supported)
"""
        return enhanced_prompt
    else:
        return base_prompt

def main():
    """Main function to generate BGP configuration with command-line options."""
    parser = argparse.ArgumentParser(
        description="Generate BGP configuration with optional enhanced security hardening."
    )
    parser.add_argument(
        "--harden-security",
        action="store_true",
        help="Add enhanced BGP security hardening features to the configuration."
    )
    
    args = parser.parse_args()
    
    prompt = generate_bgp_config(args.harden_security)
    
    response_directed = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    print("Directed Response:")
    if args.harden_security:
        print("(With Enhanced Security Hardening)")
    print("=" * 60)
    print(response_directed.choices[0].message.content)

if __name__ == "__main__":
    main()
