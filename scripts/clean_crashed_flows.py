#!/usr/bin/env python3
"""
Script pour supprimer tous les flow runs CRASHED de Prefect
"""
import asyncio
from prefect.client.orchestration import get_client
from prefect.client.schemas.filters import FlowRunFilter
from prefect.client.schemas.objects import StateType

async def delete_crashed_flows():
    """Supprime tous les flow runs avec état CRASHED"""
    async with get_client() as client:
        print("🔍 Recherche des flow runs CRASHED...")

        # Filtrer les flow runs CRASHED
        flow_runs = await client.read_flow_runs(
            flow_run_filter=FlowRunFilter(
                state={"type": {"any_": [StateType.CRASHED, StateType.FAILED, StateType.CANCELLED]}}
            ),
            limit=200
        )

        print(f"📊 Trouvé {len(flow_runs)} flow runs à supprimer")

        if not flow_runs:
            print("✅ Aucun flow run CRASHED/FAILED/CANCELLED à supprimer")
            return

        # Supprimer chaque flow run
        deleted_count = 0
        for flow_run in flow_runs:
            try:
                await client.delete_flow_run(flow_run.id)
                deleted_count += 1
                print(f"  ✓ Supprimé: {flow_run.name} ({flow_run.state.type.value})")
            except Exception as e:
                print(f"  ✗ Erreur suppression {flow_run.name}: {e}")

        print(f"\n✅ Suppression terminée: {deleted_count}/{len(flow_runs)} flow runs supprimés")

if __name__ == "__main__":
    asyncio.run(delete_crashed_flows())
