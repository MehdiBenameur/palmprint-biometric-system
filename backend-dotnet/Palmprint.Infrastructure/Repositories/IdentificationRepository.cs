using Microsoft.EntityFrameworkCore;
using Palmprint.Application.Interfaces;
using Palmprint.Domain.Entities;
using Palmprint.Infrastructure.Persistence;

namespace Palmprint.Infrastructure.Repositories;

public class IdentificationRepository : IIdentificationRepository
{
    private readonly PalmprintDbContext _context;

    public IdentificationRepository(PalmprintDbContext context)
    {
        _context = context;
    }

    public async Task<List<PalmTemplate>> GetActiveTemplatesByTenantAsync(Guid tenantId)
    {
        return await _context.PalmTemplates
            .Include(x => x.User)
            .Where(x => x.TenantId == tenantId && x.IsActive)
            .ToListAsync();
    }
}